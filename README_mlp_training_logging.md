# Training KKBOX dataset with MLP frameworks

This note is used to record the hyper-parameter tuning and model performance when we train a Multi-layer Perceptron (MLP) on the KKBOX music recommendation data.

The raw training and testing data come with seven and two million records separately, with the data grain at "user by song" level. The data also has a songs and a members table that we used to merge with the final feature set. The feature set, stored in a SQLite database locally, consists of **130** regularized numerical features. We created a data streaming iterator that goes in to the SQLite database and fetches mini-batches of the shape **batch_size by 130** with either randomized or sequential indexing. The target output of the training data is a binary variable of values **0** and **1**, representing songs not repeated by users and songs repeated by users. The model is then tested on a testing set without the output label on the [official competition page](https://www.kaggle.com/c/kkbox-music-recommendation-challenge).

## 2-Layer MLP training with 20 hidden neurons
Our initial model is a simple 2-layer framework that takes in the 130 features, applies linear transformation with 20 neurons in the hidden layer with a Tanh() activation function, and outputs the log probability of each class (0 and 1) with a LogSoftmax activation function. The model is then evaluated with a NLLLoss() loss function and back propagated with a defined learning rate and momentum. Below is the structure of this model:

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(130, 20)
        self.t1 = nn.Tanh()
        self.l2 = nn.Linear(20, 2)
        self.t2 = nn.LogSoftmax()

    def forward(self, x):
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x
```
The batch sampling process with the same random root, so the model should be reliably repeatable if needed.

### learning_rate: 0.2, momentum: 0.9, epoch: 1
Submission is made after 1 epoch of this basic model, and achieved an AUC score of 0.47037 on Kaggle leaderboard.

### learning_rate: 1, momentum: 0.5, epoch: 1
Submission achieved an AUC score of 0.47050 on Kaggle leaderboard.

### learning_rate: 1, momentum: 0.9, epoch: 2
Submission achieved an AUC score of 0.47062 on Kaggle leaderboard.


## 3 Layer MLP training with Dropout Layers
The validation accuracy scores displayed in shell output indicates that the model comes to a somewhat converged state pretty early during the first and second epoch. Considering the high number of rows in this dataset, it is possible to assume that the shallow model does not have enough parameters to catch the dynamic of the 130 features with the target output. Therefore, in our second and more complicated model, we build a framework with more layers and more neurons, together with dropout functions in each layer to ensure the model does not "remember" the input features too quickly.

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(130, 150)
        self.t1 = nn.ReLU()
        self.d1 = nn.Dropout(p=0.6)
        self.l2 = nn.Linear(150, 50)
        self.t2 = nn.ReLU()
        self.d2 = nn.Dropout(p=0.2)
        self.l3 = nn.Linear(50, 2)
        self.t3 = nn.LogSoftmax()

    def forward(self, x):
        x = self.d1(self.t1(self.l1(x)))
        x = self.d2(self.t2(self.l2(x)))
        x = self.t3(self.l3(x))
        return x
```

### learning_rate: 0.1, momentum: 0.9, epoch: 1
Submission achieved an AUC score of 0.47078 on Kaggle leaderboard.

### learning_rate: 0.1, momentum: 0.9, epoch: 2
Submission achieved an AUC score of 0.47071 on Kaggle leaderboard.

### learning_rate: 0.1, momentum: 0.9, epoch: 3
Submission achieved an AUC score of 0.47071 on Kaggle leaderboard.

The deeper neural net did not achieve a better results in the ROC-AUC, probably because the neural network is trained on post-processed data: our dimensionality reductions on the song's variables might have diluted the input data with too many variables. Moreover, the matrix factorization stage conducted before the neural network training (which only covers 7% of the total variance) is not organically connected to the training and back-propagation of the neural net, hence making the model not effective.

## Suggested Next Steps for the Project
The above results indicates that a MLP network built with pre-processed feature set is not ideal for the problem, since the feature engineering process has lost too much information and the MLP model won't be able to "learn" from the input. For next steps along the neural network, a.k.a. deep learning, approach, we will focus on creating vector embedding layers for an organic dimensionality reduction process to make the machine learn as much information from the input data as possible.
