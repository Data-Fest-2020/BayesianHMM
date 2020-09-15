import hiddensmmodel
from matplotlib import pyplot as plt
import pandas as pd
import os


def map_states(model, y_hat_test, y_true_train, smooth=True, windowSize=199):
    '''
    this instance is useful to map the predicted labels to the true ones
    as the hsmm model is discovering the number of the labels by itself,
    we have to search for the most matches and rename the labels to the true
    ones according to the train set
    '''
    # get predictet and true labels of the train set
    y_hat_train = model.states
    y_hat_train = y_hat_train[0]
    y_hat_train = pd.DataFrame(y_hat_train)
    y_true_train = pd.DataFrame(y_true_train)
    y_hat_test = pd.DataFrame(y_hat_test)
    y_true_train = y_true_train.rename(columns={0: 'y_true'})
    y_hat_train = y_hat_train.rename(columns={0: 'y_hat'})
    # add huge number to the predicted labels of the train set as the function fails
    # if the predicted ones matches with the true labels
    y_hat_train = y_hat_train + 1500
    y_hat_test = y_hat_test + 1500
    y_train = pd.DataFrame()
    y_train = pd.concat([y_true_train, y_hat_train], axis=1)
    # add auxiliary variable the the data frame in order to count the matches
    y_train['count'] = 1
    # search for the matches
    y_manC = y_train.groupby(['y_true', 'y_hat']).count()
    y_manC = y_manC.reset_index()
    states_hat = y_manC['y_hat'].unique()
    # rename the predicted labels of the train set to the true ones with the most matches
    for state_hat in states_hat:
        idx = y_manC['y_hat'][y_manC['y_hat'] == state_hat]
        idx = idx.reset_index()
        help1 = y_manC.iloc[idx['index'], :]
        maxS = help1.iloc[help1['count'].argmax()]
        y_hat_test = y_hat_test.replace({maxS['y_hat']: maxS['y_true']})
        # smooth data by taking the median of the windows and drop NANs
    if smooth == True:
        y_hat_test = y_hat_test.rolling(window=windowSize).quantile(quantile=0.65,
                                                                    interpolation="nearest")  # .dropna(axis = 0)
    # return the manipulated predicted labels
    return y_hat_test


def visualize(data):

    plt.plot(data, 'bo', markersize=0.5)
    plt.show()


def main():
    # load data (and smooth it)
    # load data
    tool = "pneumatic_screwdriver"
    # tool = "electric_screwdriver"
    # tool = "pneumatic_rivet_gun"
    smooth = False

    load_filename = tool + "_" + str(smooth) + "_all_campaigns.pkl"
    load_file_path = os.path.join(tool, load_filename)
    data = pd.read_pickle(load_file_path)

    labels = data["label"]
    timesteps = data["time [s]"]
    data.drop(["time [s]", "label"], axis=1, inplace=True)

    # define autoencoder here


    # get the labels of the training set and testing set
    y_true_train = labels[0:170000]
    y_true_test = labels[170001:]

    # training
    model = hiddensmmodel.Hsmm(data[0:170000], Nmax=15, trunc=200)

    # prediction
    y_hat_test_un = model.predict(data[170001:])

    # match the predicted labels to the true ones and smooth outliers out
    y_hat_test = map_states(model, y_hat_test_un, y_true_train,
                                              smooth=True, windowSize=19)

    # save the predictions for evaluation
    tool_results = pd.DataFrame({"y_hat": y_hat_test, "y_true": y_true_test})
    results_file = tool + "_predictions.csv"
    results_path = os.path.join(tool, results_file)
    tool_results.to_csv(results_path)

    # visualizations
    visualize(y_hat_test_un[1:5000])
    visualize(y_true_test[1:5000])
    visualize(y_hat_test[1:5000])
    visualize(y_true_test[10000:15000])
    visualize(y_hat_test[10000:15000])
    visualize(y_true_test[15000:20000])
    visualize(y_hat_test[15000:20000])
    visualize(y_true_test)
    visualize(y_hat_test)

if __name__ == "__main__":
    main()
