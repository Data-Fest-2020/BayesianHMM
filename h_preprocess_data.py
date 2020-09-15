import pandas as pd
import os.path

from datatools import MeasurementDataReader, Measurement
from datatools import Tool, DataTypes
from datatools import ACC, GYR, MAG, MIC


def decimator_over_timestep(data_to_mod, timestep, sensor):

    data_to_merge = data_to_mod.loc[data_to_mod['time [s]'] <= timestep]

    if len(data_to_merge) is not 0:

        data_to_mod = pd.merge(data_to_mod, data_to_merge, how='outer', indicator=True)

        if sensor == "mic":
            data_to_mod = data_to_mod.loc[data_to_mod._merge == 'left_only',
                                          ['time [s]', 'amplitude', 'label']]
            data_updated = {"time [s]": timestep,
                            "amplitude": round(data_to_merge["amplitude"].mean(), 4),
                            "label": int(data_to_merge["label"].mode().iloc[0])}

        else:
            data_to_mod = data_to_mod.loc[data_to_mod._merge == 'left_only',
                                          ['time [s]', 'magnetic field x-axis [T]',
                                           'magnetic field y-axis [T]', 'magnetic field z-axis [T]', 'label']]
            data_updated = {"time [s]": timestep,
                            "magnetic field x-axis [T]": data_to_merge["magnetic field x-axis [T]"].mean(),
                            "magnetic field y-axis [T]": data_to_merge["magnetic field y-axis [T]"].mean(),
                            "magnetic field z-axis [T]": data_to_merge["magnetic field z-axis [T]"].mean(),
                            "label": int(data_to_merge["label"].mode().iloc[0])}
    else:
        # defined for sensor mag for now as it contains missing timesteps
        data_updated = {"time [s]": timestep,
                        "magnetic field x-axis [T]": None,
                        "magnetic field y-axis [T]": None,
                        "magnetic field z-axis [T]": None,
                        "label": None}

    index = data_to_mod.first_valid_index()
    return index, data_updated


def decimator_over_data(data_to_mod, base_data, sensor):

    data_modified = pd.DataFrame()
    i = 0
    min_row_index = 0
    max_row_index = 100

    while i < len(base_data):

        timestep = base_data["time [s]"][i]
        to_modify_set = data_to_mod[min_row_index:max_row_index].reset_index()

        index, modified_set = decimator_over_timestep(to_modify_set, timestep, sensor)
        data_modified = data_modified.append(modified_set, ignore_index=True)

        # update row range for next iteration
        if index == None:
            min_row_index = max_row_index + 1
        else:
            min_row_index = index

        max_row_index = min_row_index + 100
        if sensor == "mag":
            max_row_index -= 80
        if max_row_index >= len(data_to_mod):
            max_row_index = len(data_to_mod)

        i += 1

        # to know the speed and progress
        if i % 5000 == 0:
            print(i)

    return data_modified


def merge_data(source, tool, smooth):

    mdr = MeasurementDataReader(source=source)
    # create a query to load Measurements or Actions
    q = mdr.query(query_type=Measurement)
    data_dict = q.filter_by(Tool == tool, DataTypes == [ACC, GYR, MAG, MIC]).get()

    all_campaigns = pd.DataFrame()

    for campaign in ["01", "02", "03", "04"]:
        data_bunch = data_dict[campaign]

        acc = pd.DataFrame(data_bunch.acc)
        gyr = pd.DataFrame(data_bunch.gyr)
        mag = pd.DataFrame(data_bunch.mag)
        mic = pd.DataFrame(data_bunch.mic)

        mic = decimator_over_data(mic, acc, "mic")
        mag = decimator_over_data(mag, acc, "mag")

        assert len(mic) == len(acc)
        assert len(mag) == len(acc)

        acc_gyr = pd.merge(acc, gyr, on=["time [s]", "label"])
        mag_mic = pd.merge(mag, mic, on=["time [s]"])
        data_combined = pd.concat([acc_gyr, mag_mic], axis=1)

        data_combined.drop(["label_x", "label_y", "time [s]"],
                           axis=1, inplace=True)
        data_combined.dropna(inplace=True)
        data_combined.reset_index(inplace=True)

        all_campaigns = all_campaigns.append(data_combined, ignore_index=True)

    all_campaigns.drop("index", axis=1, inplace=True)

    if smooth == True:
        all_campaigns = all_campaigns.rolling(window=2000).mean().dropna().reset_index().drop("index", axis=1)

    save_filename = tool + "_" + str(smooth) + "_all_campaigns.pkl"
    save_file_path = os.path.join(tool, save_filename)
    all_campaigns.to_pickle(save_file_path)

    return all_campaigns


def main():

    # path to the data
    source = "tool-tracking/tool-tracking-data/"

    mdr = MeasurementDataReader(source=source)
    # create a query to load Measurements or Actions
    q = mdr.query(query_type=Measurement)

    my_tool = "electric_screwdriver"

    data_dict = q.filter_by(Tool == my_tool, DataTypes == [ACC, GYR, MAG, MIC]).get()

    for campaign in ["01"]:

        data_bunch = data_dict[campaign]

        acc_orig = pd.DataFrame(data_bunch.acc)
        mag_orig = pd.DataFrame(data_bunch.mag)
        mic_orig = pd.DataFrame(data_bunch.mic)

        mic_modified = decimator_over_data(mic_orig, acc_orig, "mic")
        # mag_modified = decimator_over_data(mag_orig, acc_orig, "mag")

        assert len(mic_modified) == len(acc_orig)
        # assert len(mag_modified) == len(acc_orig)

        mic_name = my_tool + "_mic_" + campaign + "_modified.csv"
        mag_name = my_tool + "_mag_" + campaign + "_modified.csv"
        mic_modified.to_csv(mic_name)
        # mag_modified.to_csv(mag_name)


if __name__ == "__main__":
    main()
