import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
from torch.autograd import Variable
import math
# import 




class LSTM_based_infoLoss(nn.Module):
    def __init__(self):
        super(LSTM_based_infoLoss, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.input_dim = self.L
        self.hidden_dim = self.L
        self.layer_dim = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )



        self.dis_g = nn.Sequential(
            nn.Linear(self.L*2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),

        )

        self.dis_l = nn.Sequential(
            nn.Conv2d(self.L+50, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),


        )

        self.dis_p = nn.Sequential(
            nn.Linear(self.L, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()

        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim*self.K*2, 1),
            nn.Sigmoid()
        )




        self.model = LSTMModel(self.input_dim, self.hidden_dim, self.layer_dim, bidirectional=True)


    def save_model(self, name):
        torch.save({'feature_extractor_part1': self.feature_extractor_part1.state_dict(), 'feature_extractor_part2': self.feature_extractor_part2.state_dict(), 'classifier': self.classifier.state_dict(), 'model': self.model.state_dict()}, name)

    def resume(self, name):
        state_dict = torch.load(name)
        self.feature_extractor_part1.load_state_dict(state_dict['feature_extractor_part1'])
        self.feature_extractor_part2.load_state_dict(state_dict['feature_extractor_part2'])
        self.classifier.load_state_dict(state_dict['classifier'])
        self.model.load_state_dict(state_dict['model'])

    def forward(self, x):
        x = x.squeeze(0)
        H_local = self.feature_extractor_part1(x)
        H = H_local.view(-1, 50 * 4 * 4)
        H_global = self.feature_extractor_part2(H)  # NxL


        outputs = self.model(H_global)
        outputs = torch.cat((outputs[:,-1,0:self.hidden_dim], outputs[:,0,self.hidden_dim:]),1)
        # outputs =outputs[:,-1,:]
        Y_prob = self.classifier(outputs)
        Y_hat = torch.ge(Y_prob, 0.5).float()



        outputs = outputs.view(-1)
        return Y_prob, Y_hat, H_local, H_global 


    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _,_ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean()

        return error, Y_hat


    def calculate_classification_error_fake(self, Y):
        Y = Y.float()
        Y = Y.cpu()
        Y_hat = torch.tensor([0.])
        error = 1. - Y_hat.eq(Y).float().mean()

        Y_hat_ = torch.tensor([1.])
        error_ = 1. - Y_hat_.eq(Y).float().mean()
        return error, error_

    def calculate_objective(self, X, Y):
        # loss = nn.MSELoss()
        # BCE = nn.BCELoss()

        Y = Y.float()
        Y_prob, _, x_local, x_global = self.forward(X)


        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli


        loss_pearson = self.infoloss(x_local, x_global)

        return neg_log_likelihood + loss_pearson

    def calculate_objective_val(self, X, Y):

        Y = Y.float()
        Y_prob, _, x_local, x_global = self.forward(X)


        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        

        return neg_log_likelihood




    def infoloss(self, x_local, x_global):


        zz1_g = torch.cat((x_global, x_global), 1)
        zz2_g = torch.cat((x_global, x_global[np.random.permutation(x_global.shape[0])]), 1)
        Ej = -F.softplus(-self.dis_g(zz1_g)).mean()
        Em = F.softplus(self.dis_g(zz2_g)).mean()
        GLOBAL = (Em - Ej)

        zz1_l = torch.cat((x_global.view(x_local.shape[0],self.L,1,1).repeat(1,1,4,4), x_local),1)
        zz2_l = torch.cat((x_global.view(x_local.shape[0],self.L,1,1).repeat(1,1,4,4), x_local[np.random.permutation(x_local.shape[0])]),1)
        Ej = -F.softplus(-self.dis_l(zz1_l)).mean()
        Em = F.softplus(self.dis_l(zz2_l)).mean()
        LOCAL = (Em - Ej)

        prior = torch.rand_like(x_global)
        term_a = torch.log(self.dis_p(prior)).mean()
        term_b = torch.log(1.0 - self.dis_p(x_global)).mean()
        PRIOR = - (term_a + term_b)
        return 0.5*GLOBAL + 1*LOCAL + 0.1*PRIOR



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim
        self.bidirectional=bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=bidirectional)
        
    
    def forward(self, x):

        x = x.unsqueeze(0)

        if self.bidirectional == True:
            h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().cuda()
            c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_().cuda()
        else:
            h0 = torch.zeros(self.layer_dim*1, x.size(0), self.hidden_dim).requires_grad_().cuda()
            c0 = torch.zeros(self.layer_dim*1, x.size(0), self.hidden_dim).requires_grad_().cuda()            
        

        out, (hn, cn) = self.lstm(x, (h0, c0))
        

        return out





