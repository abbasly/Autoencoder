import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

def init_weights(m):
        if type(m) == nn.Linear:
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=10):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, latent_dim)
        )

        # self.encoder.apply(init_weights)

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            # 
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################


        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    
            
    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=10, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, output_size),
            # nn.Sigmoid()  # Sigmoid is often used to scale values between 0 and 1 for image data
        )
        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################


        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        # Feed the input image to the encoder to generate the latent vector
        latent_vector = self.encoder(x)
        # Decode the latent vector to get the reconstruction of the input
        reconstruction = self.decoder(latent_vector)
        return reconstruction

    def set_optimizer(self):
        # Define optimizer
        learning_rate = self.hparams.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        self.train()
        self = self.to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

        X = batch
        X = X.to(self.device)
        flattened_X = X.view(X.shape[0], -1)

        # Forward pass
        reconstruction = self.forward(flattened_X)

        # Compute the loss
        loss = loss_func(reconstruction, flattened_X)

        # Backward pass
        loss.backward()

        # Update parameters
        self.optimizer.step()

        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        self.eval()
        self = self.to(self.device)

        with torch.no_grad():
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)

            # Forward pass
            reconstruction = self.forward(flattened_X)

            # Compute the loss
            loss = loss_func(reconstruction, flattened_X)

        return loss

    def getReconstructions(self, loader=None):
        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)

class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Sequential(
            nn.Linear(encoder.latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################


        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        learning_rate = self.hparams.get("learning_rate", 0.001)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################


        # pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
