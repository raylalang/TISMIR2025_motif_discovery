import shutil
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from MidiBERT.finetune_model import TokenClassification, SequenceClassification


class FinetuneTrainer:
    def __init__(
        self,
        midibert,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        layer,
        lr,
        class_num,
        hs,
        testset_shape,
        cpu,
        cuda_devices=None,
        model=None,
        SeqClass=False,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not cpu else "cpu"
        )
        print("   device:", self.device)
        self.midibert = midibert
        self.SeqClass = SeqClass
        self.layer = layer
        self.cpu = cpu

        if model != None:  # load model
            print("load a fine-tuned model")
            self.model = model.to(self.device)
        else:
            print("init a fine-tune model, sequence-level task?", SeqClass)
            if SeqClass:
                self.model = SequenceClassification(self.midibert, class_num, hs).to(
                    self.device
                )
            else:
                self.model = TokenClassification(self.midibert, class_num, hs).to(
                    self.device
                )

        if torch.cuda.device_count() > 1 and not cpu:
            print(f"Using GPU {cuda_devices}")
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader

        self.optim = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.loss_func = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 2.0, 1.0, 1.0]).to(self.device), reduction="none"
        )

        self.testset_shape = testset_shape

    def compute_loss(self, predict, target, loss_mask, seq):
        loss = self.loss_func(predict, target)
        if not seq:
            loss = loss * loss_mask
            loss = torch.sum(loss) / torch.sum(loss_mask)
        else:
            loss = torch.sum(loss) / loss.shape[0]
        return loss

    def train(self):
        self.model.train()
        train_loss, train_acc = self.iteration(self.train_data, 0, self.SeqClass)
        return train_loss, train_acc

    def valid(self):
        self.model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = self.iteration(self.valid_data, 1, self.SeqClass)
        return valid_loss, valid_acc

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_loss, test_acc, all_output = self.iteration(
                self.test_data, 2, self.SeqClass
            )
        return test_loss, test_acc, all_output

    def iteration(self, training_data, mode, seq):
        pbar = tqdm.tqdm(training_data, disable=False)
        # print (len(training_data))

        total_acc, total_cnt, total_loss = 0, 0, 0
        confusion = [[0, 0], [0, 0]]

        if mode == 2:  # testing
            all_output = torch.empty(self.testset_shape)
            cnt = 0

        batch_count = 0
        for x, y in pbar:  # (batch, 512, 768)
            batch = x.shape[0]

            batch_count += 1
            # if mode == 0 and batch_count > len(training_data) // 2:
            #     break

            x, y = x.to(self.device), y.to(
                self.device
            )  # seq: (batch, 512, 4), (batch) / token: , (batch, 512)
            # print (x.shape, y.shape)
            # avoid attend to pad word
            if not seq:
                attn = (y != 0).float().to(self.device)  # (batch,512)
            else:
                attn = torch.ones((batch, 512)).to(self.device)  # attend each of them

            y_hat = self.model.forward(
                x, attn, self.layer
            )  # seq: (batch, class_num) / token: (batch, 512, class_num)

            # get the most likely choice with max
            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)
            if mode == 2:
                all_output[cnt : cnt + batch] = output
                cnt += batch

            # accuracy
            if not seq:
                acc = torch.sum((y == output).float() * attn)
                total_acc += acc
                total_cnt += torch.sum(attn).item()
                # print (y)
                # print (output)
                confusion[0][0] += int(
                    torch.sum(((y == 3) & (output != 1)).float() * attn)
                )
                confusion[0][1] += int(
                    torch.sum(((y == 3) & (output == 1)).float() * attn)
                )
                confusion[1][0] += int(
                    torch.sum(((y == 1) & (output != 1)).float() * attn)
                )
                confusion[1][1] += int(
                    torch.sum(((y == 1) & (output == 1)).float() * attn)
                )
            else:
                acc = torch.sum((y == output).float())
                total_acc += acc
                total_cnt += y.shape[0]

            # calculate losses
            if not seq:
                y_hat = y_hat.permute(0, 2, 1)
            loss = self.compute_loss(y_hat, y, attn, seq)
            total_loss += loss.item()

            # udpate only in train
            if mode == 0:
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

        total_notes = (
            confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]
        )
        test_accuracy = (confusion[0][0] + confusion[1][1]) / total_notes
        test_precision = (confusion[1][1]) / max(
            (confusion[0][1] + confusion[1][1]), 0.0001
        )
        test_recall = (confusion[1][1]) / max(
            (confusion[1][0] + confusion[1][1]), 0.0001
        )
        test_f1 = (2.0 * test_precision * test_recall) / max(
            (test_precision + test_recall), 0.0001
        )
        print(confusion)

        # if mode == 2:
        #     return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4), all_output
        # return round(total_loss/len(training_data),4), round(total_acc.item()/total_cnt,4)
        if mode == 2:
            return (
                round(total_loss / len(training_data), 4),
                round(test_f1, 4),
                all_output,
            )
        return round(total_loss / len(training_data), 4), round(test_f1, 4)

    def save_checkpoint(
        self, epoch, train_acc, valid_acc, valid_loss, train_loss, is_best, filename
    ):
        if torch.cuda.device_count() > 1 and not self.cpu:
            state = {
                "epoch": epoch + 1,
                "state_dict": self.model.module.state_dict(),
                "valid_acc": valid_acc,
                "valid_loss": valid_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "optimizer": self.optim.state_dict(),
            }
        else:
            state = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "valid_acc": valid_acc,
                "valid_loss": valid_loss,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "optimizer": self.optim.state_dict(),
            }

        torch.save(state, filename)

        best_mdl = filename.replace("model.ckpt", "model_best.ckpt")
        print(f"Saving model to {filename} ...")
        if is_best:
            shutil.copyfile(filename, best_mdl)
            print(f"Saving model to {best_mdl} ...")
