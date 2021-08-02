from datetime import datetime

import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)


def ff64(self):
    return np.array(self, np.float)

class Brain:
    def __init__(self, **kwargs):
        #self._model = self._mount_model()
        self.logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir, histogram_freq=1)
        self._model = self.___mount_model()


    def predict(self, playerIndex, no_parsed_sequences):
        formatted_playerSelection = \
            ff64([[playerIndex["PlayerID"], playerIndex["EnemyID"]] for x in range(len(no_parsed_sequences))])
        formatted_injuries = ff64([x._values.injuries for x in no_parsed_sequences])

        formatted_limbPositionsYou = ff64([x._values.limb_positions[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))
        formatted_limbPositionsEnemy = ff64([x._values.limb_positions[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))

        formatted_limbVelocitiesYou = ff64([x._values.limb_velocities[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))
        formatted_limbVelocitiesEnemy = ff64([x._values.limb_velocities[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))

        formatted_jointStatesYou = ff64([x._values.joint_states[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 22))
        formatted_jointStatesEnemy = ff64([x._values.joint_states[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 22))

        formatted_groinRotationsYou = ff64([x._values.groin_rotations[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 16))
        formatted_groinRotationsEnemy = ff64([x._values.groin_rotations[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 16))

        prediction = self._model.predict(
            {
                "playerSelection": ff64([formatted_playerSelection]),
                "injuries": ff64([formatted_injuries]),
                "limbPositionsYou": ff64([formatted_limbPositionsYou]),
                "limbPositionsEnemy": ff64([formatted_limbPositionsEnemy]),
                "limbVelocitiesYou": ff64([formatted_limbVelocitiesYou]),
                "limbVelocitiesEnemy": ff64([formatted_limbVelocitiesEnemy]),
                "jointStatesYou": ff64([formatted_jointStatesYou]),
                "jointStatesEnemy": ff64([formatted_jointStatesEnemy]),
                "groinRotationsYou": ff64([formatted_groinRotationsYou]),
                "groinRotationsEnemy": ff64([formatted_groinRotationsEnemy]),
            },
            1
        )
        return prediction

    def predict2(self, playerIndex, no_parsed_sequences):
        formatted_playerSelection = \
            ff64([[playerIndex["PlayerID"], playerIndex["EnemyID"]] for x in range(len(no_parsed_sequences))])
        formatted_injuries = ff64([x._values.injuries for x in no_parsed_sequences])

        formatted_limbPositionsYou = ff64([x._values.limb_positions[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))
        formatted_limbPositionsEnemy = ff64([x._values.limb_positions[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))

        formatted_limbVelocitiesYou = ff64([x._values.limb_velocities[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))
        formatted_limbVelocitiesEnemy = ff64([x._values.limb_velocities[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))

        formatted_jointStatesYou = ff64([x._values.joint_states[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 22))
        formatted_jointStatesEnemy = ff64([x._values.joint_states[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 22))

        formatted_groinRotationsYou = ff64([x._values.groin_rotations[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 16))
        formatted_groinRotationsEnemy = ff64([x._values.groin_rotations[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 16))

        prediction = self._model.predict(
            {
                "limbPositionsYou": ff64([[formatted_limbPositionsYou[x] if x < len(formatted_limbPositionsYou) else [0 for _ in range(len(formatted_limbPositionsYou[0]))] for x in range(15)]]),
                "limbPositionsEnemy": ff64([[formatted_limbPositionsEnemy[x] if x < len(formatted_limbPositionsEnemy) else [0 for _ in range(len(formatted_limbPositionsEnemy[0]))] for x in range(15)]]),
                "limbVelocitiesYou": ff64([[formatted_limbVelocitiesYou[x] if x < len(formatted_limbVelocitiesYou) else [0 for _ in range(len(formatted_limbVelocitiesYou[0]))] for x in range(15)]]),
                "limbVelocitiesEnemy": ff64([[formatted_limbVelocitiesEnemy[x] if x < len(formatted_limbVelocitiesEnemy) else [0 for _ in range(len(formatted_limbVelocitiesEnemy[0]))] for x in range(15)]]),
                "jointStatesYou": ff64([[formatted_jointStatesYou[x] if x < len(formatted_jointStatesYou) else [0 for _ in range(len(formatted_jointStatesYou[0]))] for x in range(15)]]),
                "jointStatesEnemy": ff64([[formatted_jointStatesEnemy[x] if x < len(formatted_jointStatesEnemy) else [0 for _ in range(len(formatted_jointStatesEnemy[0]))] for x in range(15)]]),
                "groinRotationsYou": ff64([[formatted_groinRotationsYou[x] if x < len(formatted_groinRotationsYou) else [0 for _ in range(len(formatted_groinRotationsYou[0]))] for x in range(15)]]),
                "groinRotationsEnemy": ff64([[formatted_groinRotationsEnemy[x] if x < len(formatted_groinRotationsEnemy) else [0 for _ in range(len(formatted_groinRotationsEnemy[0]))] for x in range(15)]]),
            },
            1
        )
        return prediction

    def convertNoDataToAiData(self, playerIndex, no_parsed_sequences):
        formatted_playerSelection = \
            ff64([[playerIndex["PlayerID"], playerIndex["EnemyID"]] for x in range(len(no_parsed_sequences))])
        formatted_injuries = ff64([x._values.injuries for x in no_parsed_sequences])

        formatted_limbPositionsYou = ff64([x._values.limb_positions[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))
        formatted_limbPositionsEnemy = ff64([x._values.limb_positions[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))

        formatted_limbVelocitiesYou = ff64([x._values.limb_velocities[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))
        formatted_limbVelocitiesEnemy = ff64([x._values.limb_velocities[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 21*3))

        formatted_jointStatesYou = ff64([x._values.joint_states[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 22))
        formatted_jointStatesEnemy = ff64([x._values.joint_states[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 22))

        formatted_groinRotationsYou = ff64([x._values.groin_rotations[playerIndex["PlayerID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 16))
        formatted_groinRotationsEnemy = ff64([x._values.groin_rotations[playerIndex["EnemyID"]] for x in no_parsed_sequences]).reshape((len(no_parsed_sequences), 16))
        return [formatted_playerSelection, formatted_injuries, formatted_limbPositionsYou, formatted_limbPositionsEnemy, formatted_limbVelocitiesYou, formatted_limbVelocitiesEnemy,
                formatted_jointStatesYou, formatted_jointStatesEnemy, formatted_groinRotationsYou, formatted_groinRotationsEnemy]

    def train(self, fbatch, encode):
        batch_x = {
            "playerSelection": [],
            "limbPositionsYou": [],
            "limbPositionsEnemy": [],
            "limbVelocitiesYou": [],
            "limbVelocitiesEnemy": [],
            "jointStatesYou": [],
            "jointStatesEnemy": [],
            "groinRotationsYou": [],
            "groinRotationsEnemy": [],
            "injuries": [],
        }


        batch_y = {
            "roPlayerSelection": [],
            "roLimbPositionsYou": [],
            "roLimbPositionsEnemy": [],
            "roLimbVelocitiesYou": [],
            "roLimbVelocitiesEnemy": [],
            "roJointStatesYou": [],
            "roJointStatesEnemy": [],
            "roGroinRotationsYou": [],
            "roGroinRotationsEnemy": [],
            "roInjuries": [],
        }


        for k in range(len(fbatch[0])):
            formatted_playerSelection = fbatch[0][k]
            formatted_injuries = fbatch[1][k]

            formatted_limbPositionsYou = fbatch[2][k]
            formatted_limbPositionsEnemy = fbatch[3][k]

            formatted_limbVelocitiesYou = fbatch[4][k]
            formatted_limbVelocitiesEnemy = fbatch[5][k]

            formatted_jointStatesYou = fbatch[6][k]
            formatted_jointStatesEnemy = fbatch[7][k]

            formatted_groinRotationsYou = fbatch[8][k]
            formatted_groinRotationsEnemy = fbatch[9][k]

            sequence_size = 10
            sa = 0
            sya = 1
            for it in range(1, len(formatted_playerSelection) - 1):
                sa = 0 if it - sequence_size < 0 else it - sequence_size
                sb = it
                sya = 1 if (it + 1) - sequence_size < 1 else (it + 1) - sequence_size
                syb = it + 1

                batch_x["playerSelection"].append(formatted_playerSelection[sa:sb])
                batch_x["injuries"].append(formatted_injuries[sa:sb])
                batch_x["limbPositionsYou"].append(formatted_limbPositionsYou[sa:sb])
                batch_x["limbPositionsEnemy"].append(formatted_limbPositionsEnemy[sa:sb])
                batch_x["limbVelocitiesYou"].append(formatted_limbVelocitiesYou[sa:sb])
                batch_x["limbVelocitiesEnemy"].append(formatted_limbVelocitiesEnemy[sa:sb])
                batch_x["jointStatesYou"].append(formatted_jointStatesYou[sa:sb])
                batch_x["jointStatesEnemy"].append(formatted_jointStatesEnemy[sa:sb])
                batch_x["groinRotationsYou"].append(formatted_groinRotationsYou[sa:sb])
                batch_x["groinRotationsEnemy"].append(formatted_groinRotationsEnemy[sa:sb])

                batch_y["roPlayerSelection"].append(formatted_playerSelection[syb])
                batch_y["roInjuries"].append(formatted_injuries[syb])
                batch_y["roLimbPositionsYou"].append(formatted_limbPositionsYou[syb])
                batch_y["roLimbPositionsEnemy"].append(formatted_limbPositionsEnemy[syb])
                batch_y["roLimbVelocitiesYou"].append(formatted_limbVelocitiesYou[syb])
                batch_y["roLimbVelocitiesEnemy"].append(formatted_limbVelocitiesEnemy[syb])
                batch_y["roJointStatesYou"].append(formatted_jointStatesYou[syb])
                batch_y["roJointStatesEnemy"].append(formatted_jointStatesEnemy[syb])
                batch_y["roGroinRotationsYou"].append(formatted_groinRotationsYou[syb])
                batch_y["roGroinRotationsEnemy"].append(formatted_groinRotationsEnemy[syb])


        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./"+encode["DirModel"]+"/"+encode["NameModel"],
            verbose=0,
            save_weights_only=True,
            save_freq=128
        )

        self._model.fit(
            {
                "playerSelection": tf.keras.preprocessing.sequence.pad_sequences(batch_x["playerSelection"], dtype="float32"),
                "limbPositionsYou": tf.keras.preprocessing.sequence.pad_sequences(batch_x["limbPositionsYou"], dtype="float32"),
                "limbPositionsEnemy": tf.keras.preprocessing.sequence.pad_sequences(batch_x["limbPositionsEnemy"], dtype="float32"),
                "limbVelocitiesYou": tf.keras.preprocessing.sequence.pad_sequences(batch_x["limbVelocitiesYou"], dtype="float32"),
                "limbVelocitiesEnemy": tf.keras.preprocessing.sequence.pad_sequences(batch_x["limbVelocitiesEnemy"], dtype="float32"),
                "jointStatesYou": tf.keras.preprocessing.sequence.pad_sequences(batch_x["jointStatesYou"], dtype="float32"),
                "jointStatesEnemy": tf.keras.preprocessing.sequence.pad_sequences(batch_x["jointStatesEnemy"], dtype="float32"),
                "groinRotationsYou": tf.keras.preprocessing.sequence.pad_sequences(batch_x["groinRotationsYou"], dtype="float32"),
                "groinRotationsEnemy": tf.keras.preprocessing.sequence.pad_sequences(batch_x["groinRotationsEnemy"], dtype="float32"),
                "injuries": tf.keras.preprocessing.sequence.pad_sequences(batch_x["injuries"], dtype="float32"),
            },
            {
                "roPlayerSelection": ff64(batch_y["roPlayerSelection"]),
                "roLimbPositionsYou": ff64(batch_y["roLimbPositionsYou"]).reshape((len(batch_y["roPlayerSelection"]), 21, 3)),
                "roLimbPositionsEnemy": ff64(batch_y["roLimbPositionsEnemy"]).reshape((len(batch_y["roPlayerSelection"]), 21, 3)),
                "roLimbVelocitiesYou": ff64(batch_y["roLimbVelocitiesYou"]).reshape((len(batch_y["roPlayerSelection"]), 21, 3)),
                "roLimbVelocitiesEnemy": ff64(batch_y["roLimbVelocitiesEnemy"]).reshape((len(batch_y["roPlayerSelection"]), 21, 3)),
                "roJointStatesYou": ff64(batch_y["roJointStatesYou"]).reshape((len(batch_y["roPlayerSelection"]), 1, 22)),
                "roJointStatesEnemy": ff64(batch_y["roJointStatesEnemy"]).reshape((len(batch_y["roPlayerSelection"]), 1, 22)),
                "roGroinRotationsYou": ff64(batch_y["roGroinRotationsYou"]).reshape((len(batch_y["roPlayerSelection"]), 4, 4)),
                "roGroinRotationsEnemy": ff64(batch_y["roGroinRotationsEnemy"]).reshape((len(batch_y["roPlayerSelection"]), 4, 4)),
                "roInjuries": ff64(batch_y["roInjuries"]),
            },
            epochs=256,
            batch_size=len(batch_y["roInjuries"]),
            verbose=False,
            callbacks=[self.tensorboard_callback, cp_callback]
        )

    def advancedTrain(self,
                      data,
                      encode,
                      batch_size=1,
                      epochs=1,
                      sequence_size=10,
                      step=1,
                      **kwargs):
        size_sample = len(data[2])

        batch_x = {
            "playerSelection": [],
            "limbPositionsYou": [],
            "limbPositionsEnemy": [],
            "limbVelocitiesYou": [],
            "limbVelocitiesEnemy": [],
            "jointStatesYou": [],
            "jointStatesEnemy": [],
            "groinRotationsYou": [],
            "groinRotationsEnemy": [],
            "injuries": [],
        }

        batch_y = {
            "roPlayerSelection": [],
            "roLimbPositionsYou": [],
            "roLimbPositionsEnemy": [],
            "roLimbVelocitiesYou": [],
            "roLimbVelocitiesEnemy": [],
            "roJointStatesYou": [],
            "roJointStatesEnemy": [],
            "roGroinRotationsYou": [],
            "roGroinRotationsEnemy": [],
            "roInjuries": [],
        }

        for k in range(size_sample):
            #formatted_playerSelection = fbatch[0][k]
            #formatted_injuries = fbatch[1][k]

            formatted_limbPositionsYou = data[2][k]
            formatted_limbPositionsEnemy = data[3][k]

            formatted_limbVelocitiesYou = data[4][k]
            formatted_limbVelocitiesEnemy = data[5][k]

            formatted_jointStatesYou = data[6][k]
            formatted_jointStatesEnemy = data[7][k]

            formatted_groinRotationsYou = data[8][k]
            formatted_groinRotationsEnemy = data[9][k]

            for it in range(1, len(formatted_limbPositionsYou) - 1):
                sa = 0 if it - sequence_size < 0 else it - sequence_size
                sb = it
                syb = it + 1

                batch_x["limbPositionsYou"].append(np.array([formatted_limbPositionsYou[sa : sb][x] if x < len(formatted_limbPositionsYou[sa : sb]) else np.array([0 for _ in range(len(formatted_limbPositionsYou[sa:sb][0]))]) for x in range(sequence_size)]))
                batch_x["limbPositionsEnemy"].append(np.array([formatted_limbPositionsEnemy[sa : sb][x] if x < len(formatted_limbPositionsEnemy[sa : sb]) else np.array([0 for _ in range(len(formatted_limbPositionsEnemy[sa:sb][0]))]) for x in range(sequence_size)]))
                batch_x["limbVelocitiesYou"].append(np.array([formatted_limbVelocitiesYou[sa : sb][x] if x < len(formatted_limbVelocitiesYou[sa : sb]) else np.array([0 for _ in range(len(formatted_limbVelocitiesYou[sa:sb][0]))]) for x in range(sequence_size)]))
                batch_x["limbVelocitiesEnemy"].append(np.array([formatted_limbVelocitiesEnemy[sa : sb][x] if x < len(formatted_limbVelocitiesEnemy[sa : sb]) else np.array([0 for _ in range(len(formatted_limbVelocitiesEnemy[sa:sb][0]))]) for x in range(sequence_size)]))
                batch_x["jointStatesYou"].append(np.array([formatted_jointStatesYou[sa : sb][x] if x < len(formatted_jointStatesYou[sa : sb]) else np.array([0 for _ in range(len(formatted_jointStatesYou[sa:sb][0]))]) for x in range(sequence_size)]))
                batch_x["jointStatesEnemy"].append(np.array([formatted_jointStatesEnemy[sa : sb][x] if x < len(formatted_jointStatesEnemy[sa : sb]) else np.array([0 for _ in range(len(formatted_jointStatesEnemy[sa:sb][0]))]) for x in range(sequence_size)]))
                batch_x["groinRotationsYou"].append(np.array([formatted_groinRotationsYou[sa : sb][x] if x < len(formatted_groinRotationsYou[sa : sb]) else np.array([0 for _ in range(len(formatted_groinRotationsYou[sa:sb][0]))]) for x in range(sequence_size)]))
                batch_x["groinRotationsEnemy"].append(np.array([formatted_groinRotationsEnemy[sa : sb][x] if x < len(formatted_groinRotationsEnemy[sa : sb]) else np.array([0 for _ in range(len(formatted_groinRotationsEnemy[sa:sb][0]))]) for x in range(sequence_size)]))

                batch_y["roLimbPositionsYou"].append(formatted_limbPositionsYou[syb])
                batch_y["roLimbPositionsEnemy"].append(formatted_limbPositionsEnemy[syb])
                batch_y["roLimbVelocitiesYou"].append(formatted_limbVelocitiesYou[syb])
                batch_y["roLimbVelocitiesEnemy"].append(formatted_limbVelocitiesEnemy[syb])
                batch_y["roJointStatesYou"].append(formatted_jointStatesYou[syb])
                batch_y["roJointStatesEnemy"].append(formatted_jointStatesEnemy[syb])
                batch_y["roGroinRotationsYou"].append(formatted_groinRotationsYou[syb])
                batch_y["roGroinRotationsEnemy"].append(formatted_groinRotationsEnemy[syb])

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath="./" + encode["DirModel"] + "/" + encode["NameModel"],
            verbose=0,
            save_weights_only=True,
            save_freq=epochs - 1
        )
        batch_len = int(len(batch_x["groinRotationsEnemy"]) / batch_size)
        print("Batch len " + str(batch_len) + " Batch size " + str(batch_size))
        for __ in range(step):
            print("Step " + str(__))
            for _ in range(batch_len):
                print("Batch Train " + str(_))
                self._model.fit(
                    {
                        "limbPositionsYou": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["limbPositionsYou"][_ : _ + batch_size], dtype="float32"),

                        "limbPositionsEnemy": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["limbPositionsEnemy"][_ : _ + batch_size], dtype="float32"),

                        "limbVelocitiesYou": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["limbVelocitiesYou"][_ : _ + batch_size], dtype="float32"),

                        "limbVelocitiesEnemy": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["limbVelocitiesEnemy"][_ : _ + batch_size], dtype="float32"),

                        "jointStatesYou": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["jointStatesYou"][_ : _ + batch_size], dtype="float32"),

                        "jointStatesEnemy": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["jointStatesEnemy"][_ : _ + batch_size], dtype="float32"),

                        "groinRotationsYou": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["groinRotationsYou"][_ : _ + batch_size], dtype="float32"),

                        "groinRotationsEnemy": tf.keras.preprocessing.sequence.pad_sequences(
                            batch_x["groinRotationsEnemy"][_ : _ + batch_size], dtype="float32"),
                    },
                    {
                        "roLimbPositionsYou": ff64(batch_y["roLimbPositionsYou"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 21, 3)),
                        "roLimbPositionsEnemy": ff64(batch_y["roLimbPositionsEnemy"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 21, 3)),
                        "roLimbVelocitiesYou": ff64(batch_y["roLimbVelocitiesYou"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 21, 3)),
                        "roLimbVelocitiesEnemy": ff64(batch_y["roLimbVelocitiesEnemy"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 21, 3)),
                        "roJointStatesYou": ff64(batch_y["roJointStatesYou"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 1, 22)),
                        "roJointStatesEnemy": ff64(batch_y["roJointStatesEnemy"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 1, 22)),
                        "roGroinRotationsYou": ff64(batch_y["roGroinRotationsYou"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 4, 4)),
                        "roGroinRotationsEnemy": ff64(batch_y["roGroinRotationsEnemy"][_ : _ + batch_size]).reshape(
                            (len(batch_y["roLimbPositionsYou"][_ : _ + batch_size]), 4, 4)),
                    },
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=False,
                    callbacks=[self.tensorboard_callback, cp_callback]
                )

    def _visualize_model(self, model, name="model.png"):
        tf.keras.utils.plot_model(model=model, to_file="./ModelSchema/" + name, show_shapes=True, show_layer_names=True)

    def _mount_model(self):
        stps = keras.Input(shape=(None, 1 * 2), name="playerSelection")
        stlp_y = keras.Input(shape=(None, 21 * 3), name="limbPositionsYou")
        stlp_e = keras.Input(shape=(None, 21 * 3), name="limbPositionsEnemy")
        stlv_y = keras.Input(shape=(None, 21 * 3), name="limbVelocitiesYou")
        stlv_e = keras.Input(shape=(None, 21 * 3), name="limbVelocitiesEnemy")
        stgr_y = keras.Input(shape=(None, 4 * 4), name="groinRotationsYou")
        stgr_e = keras.Input(shape=(None, 4 * 4), name="groinRotationsEnemy")
        stjs_y = keras.Input(shape=(None, 1 * 22), name="jointStatesYou")
        stjs_e = keras.Input(shape=(None, 1 * 22), name="jointStatesEnemy")
        svi = keras.Input(shape=(None, 1 * 2), name="injuries")

        stps_features = layers.LSTM(8, return_sequences=False, activation="sigmoid")(stps)
        stlp_y_features = layers.LSTM(32, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlp_y)
        stlp_e_features = layers.LSTM(32, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlp_e)
        stlv_y_features = layers.LSTM(64, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlv_y)
        stlv_e_features = layers.LSTM(64, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlv_e)
        stgr_y_features = layers.LSTM(32, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stgr_y)
        stgr_e_features = layers.LSTM(32, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stgr_e)
        stjs_y_features = layers.LSTM(256, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stjs_y)
        stjs_e_features = layers.LSTM(256, return_sequences=False, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stjs_e)
        svi_features = layers.LSTM(8, return_sequences=False, activation="sigmoid")(svi)

        x = layers.concatenate([stlp_y_features, stlp_e_features, stlv_y_features,
                                stlv_e_features, stgr_y_features, stgr_e_features,
                                stps_features, stjs_y_features, stjs_e_features, svi_features])

        x = layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)

        _olpy = layers.Dense(63, name="oLimbPositionsYou")(x)
        olpy = layers.Reshape((21, 3), name="roLimbPositionsYou")(_olpy)
        _olpe = layers.Dense(63, name="oLimbPositionsEnemy")(x)
        olpe = layers.Reshape((21, 3), name="roLimbPositionsEnemy")(_olpe)
        _olvy = layers.Dense(63, name="oLimbVelocitiesYou")(x)
        olvy = layers.Reshape((21, 3), name="roLimbVelocitiesYou")(_olvy)
        _olve = layers.Dense(63, name="oLimbVelocitiesEnemy")(x)
        olve = layers.Reshape((21, 3), name="roLimbVelocitiesEnemy")(_olve)
        _oory = layers.Dense(16, name="oGroinRotationsYou")(x)
        ogry = layers.Reshape((4, 4), name="roGroinRotationsYou")(_oory)
        _oore = layers.Dense(16, name="oGroinRotationsEnemy")(x)
        ogre= layers.Reshape((4, 4), name="roGroinRotationsEnemy")(_oore)
        _otps = layers.Dense(2, name="oPlayerSelection")(x)
        otps= layers.Reshape((1, 2), name="roPlayerSelection")(_otps)
        _x = layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(x)
        _osvi = layers.Dense(2, name="oInjuries")(_x)
        osvi = layers.Reshape((1, 2), name="roInjuries")(_osvi)

        __x = layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)

        _ojsy = layers.Dense(22, name="oJointStatesYou")(__x)
        ojsy = layers.Reshape((1, 22), name="roJointStatesYou")(_ojsy)
        _ojse = layers.Dense(22, name="oJointStatesEnemy")(__x)
        ojse = layers.Reshape((1, 22), name="roJointStatesEnemy")(_ojse)

        model = keras.Model(
            inputs=[stlp_y, stlp_e, stlv_y, stlv_e, stgr_y, stgr_e, stps, svi, stjs_y, stjs_e],
            outputs=[olpy, olpe, olvy, olve, ogry, ogre, otps, osvi, ojsy, ojse],
        )
        model.compile(
            optimizer=keras.optimizers.RMSprop(1e-3),
            loss={
                "roLimbPositionsYou": keras.losses.MeanAbsoluteError(),
                "roLimbPositionsEnemy": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesYou": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesEnemy": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsYou": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsEnemy": keras.losses.MeanAbsoluteError(),
                "roPlayerSelection": keras.losses.BinaryCrossentropy(),
                "roInjuries": keras.losses.MeanAbsoluteError(),
                "roJointStatesYou": keras.losses.MeanAbsoluteError(),
                "roJointStatesEnemy": keras.losses.MeanAbsoluteError(),
            }
        )
        model.summary()
        self._visualize_model(model, name="modelStandard.png")
        model.load_weights("./Data/Model.ckpt")
        return model


    def __mount_model(self):
        stps = keras.Input(shape=(None, 1 * 2), name="playerSelection")
        stlp_y = keras.Input(shape=(None, 21 * 3), name="limbPositionsYou")
        stlp_e = keras.Input(shape=(None, 21 * 3), name="limbPositionsEnemy")
        stlv_y = keras.Input(shape=(None, 21 * 3), name="limbVelocitiesYou")
        stlv_e = keras.Input(shape=(None, 21 * 3), name="limbVelocitiesEnemy")
        stgr_y = keras.Input(shape=(None, 4 * 4), name="groinRotationsYou")
        stgr_e = keras.Input(shape=(None, 4 * 4), name="groinRotationsEnemy")
        stjs_y = keras.Input(shape=(None, 1 * 22), name="jointStatesYou")
        stjs_e = keras.Input(shape=(None, 1 * 22), name="jointStatesEnemy")
        svi = keras.Input(shape=(None, 1 * 2), name="injuries")

        stps_features = layers.LSTM(8, return_sequences=True, activation="sigmoid")(stps)
        stlp_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlp_y)
        stlp_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlp_e)
        stlv_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlv_y)
        stlv_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlv_e)
        stgr_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stgr_y)
        stgr_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stgr_e)
        stjs_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stjs_y)
        stjs_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stjs_e)
        svi_features = layers.LSTM(8, return_sequences=True, activation="sigmoid")(svi)

        stps_features = layers.LSTM(8, return_sequences=True, activation="sigmoid")(stps_features)
        stlp_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlp_y_features)
        stlp_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlp_e_features)
        stlv_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlv_y_features)
        stlv_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stlv_e_features)
        stgr_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stgr_y_features)
        stgr_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stgr_e_features)
        stjs_y_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stjs_y_features)
        stjs_e_features = layers.LSTM(16, return_sequences=True, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(stjs_e_features)
        svi_features = layers.LSTM(8, return_sequences=True, activation="sigmoid")(svi_features)

        x = layers.concatenate([stlp_y_features, stlp_e_features, stlv_y_features,
                                stlv_e_features, stgr_y_features, stgr_e_features,
                                stps_features, stjs_y_features, stjs_e_features, svi_features])

        #x = layers.Conv1D(64, 2)(x)
        #x = layers.MaxPool1D()(x)
        #x = layers.Conv1D(32, 2)(x)
        #x = layers.MaxPool1D()(x)
        #x = layers.Conv1D(32, 2)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(alpha=0.1))(x)

        _olpy = layers.Dense(63, name="oLimbPositionsYou")(x)
        olpy = layers.Reshape((21, 3), name="roLimbPositionsYou")(_olpy)
        _olpe = layers.Dense(63, name="oLimbPositionsEnemy")(x)
        olpe = layers.Reshape((21, 3), name="roLimbPositionsEnemy")(_olpe)
        _olvy = layers.Dense(63, name="oLimbVelocitiesYou")(x)
        olvy = layers.Reshape((21, 3), name="roLimbVelocitiesYou")(_olvy)
        _olve = layers.Dense(63, name="oLimbVelocitiesEnemy")(x)
        olve = layers.Reshape((21, 3), name="roLimbVelocitiesEnemy")(_olve)
        _oory = layers.Dense(16, name="oGroinRotationsYou")(x)
        ogry = layers.Reshape((4, 4), name="roGroinRotationsYou")(_oory)
        _oore = layers.Dense(16, name="oGroinRotationsEnemy")(x)
        ogre= layers.Reshape((4, 4), name="roGroinRotationsEnemy")(_oore)
        _otps = layers.Dense(2, name="oPlayerSelection")(x)
        otps= layers.Reshape((1, 2), name="roPlayerSelection")(_otps)
        _osvi = layers.Dense(2, name="oInjuries")(x)
        osvi = layers.Reshape((1, 2), name="roInjuries")(_osvi)
        _ojsy = layers.Dense(22, name="oJointStatesYou")(x)
        ojsy = layers.Reshape((1, 22), name="roJointStatesYou")(_ojsy)
        _ojse = layers.Dense(22, name="oJointStatesEnemy")(x)
        ojse = layers.Reshape((1, 22), name="roJointStatesEnemy")(_ojse)

        model = keras.Model(
            inputs=[stlp_y, stlp_e, stlv_y, stlv_e, stgr_y, stgr_e, stps, svi, stjs_y, stjs_e],
            outputs=[olpy, olpe, olvy, olve, ogry, ogre, otps, osvi, ojsy, ojse],
        )
        model.compile(
            optimizer=keras.optimizers.Adadelta(1e-3),
            loss={
                "roLimbPositionsYou": keras.losses.MeanAbsoluteError(),
                "roLimbPositionsEnemy": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesYou": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesEnemy": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsYou": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsEnemy": keras.losses.MeanAbsoluteError(),
                "roPlayerSelection": keras.losses.BinaryCrossentropy(),
                "roInjuries": keras.losses.MeanAbsoluteError(),
                "roJointStatesYou": keras.losses.MeanAbsoluteError(),
                "roJointStatesEnemy": keras.losses.MeanAbsoluteError(),
            },
            metrics={
                "roPlayerSelection": keras.metrics.BinaryAccuracy(),
            }
        )
        model.summary()
        self._visualize_model(model, name="modelAdvanced.png")
        model.load_weights("./Datav3/Model.ckpt")
        return model



    def ___mount_model(self):
        stlp_y = keras.Input(shape=(15, 63), name="limbPositionsYou")
        stlp_e = keras.Input(shape=(15, 63), name="limbPositionsEnemy")
        stlv_y = keras.Input(shape=(15, 63), name="limbVelocitiesYou")
        stlv_e = keras.Input(shape=(15, 63), name="limbVelocitiesEnemy")
        stgr_y = keras.Input(shape=(15, 16), name="groinRotationsYou")
        stgr_e = keras.Input(shape=(15, 16), name="groinRotationsEnemy")
        stjs_y = keras.Input(shape=(15, 22), name="jointStatesYou")
        stjs_e = keras.Input(shape=(15, 22), name="jointStatesEnemy")

        dbrain = 256

        sstlp_e = layers.Dense(dbrain, activation="swish")(stlp_e)
        sstlv_e = layers.Dense(dbrain, activation="swish")(stlv_e)
        sstgr_e = layers.Dense(dbrain, activation="swish")(stgr_e)
        sstjs_e = layers.Dense(dbrain, activation="swish")(stjs_e)

        sstlp_e = layers.Reshape((1, 15, dbrain))(sstlp_e)
        sstlv_e = layers.Reshape((1, 15, dbrain))(sstlv_e)
        sstgr_e = layers.Reshape((1, 15, dbrain))(sstgr_e)
        sstjs_e = layers.Reshape((1, 15, dbrain))(sstjs_e)

        concat_a = layers.concatenate([sstlp_e, sstlv_e, sstgr_e, sstjs_e], axis=1)

        dense_a = layers.ConvLSTM1D(filters=512, kernel_size=15, activation="swish", return_sequences=False, dropout=0.01, recurrent_dropout=0.01)(concat_a)

        _olpe = layers.Dense(63, name="oLimbPositionsEnemy")(dense_a)
        olpe = layers.Reshape((21, 3), name="roLimbPositionsEnemy")(_olpe)

        _olve = layers.Dense(63, name="oLimbVelocitiesEnemy")(dense_a)
        olve = layers.Reshape((21, 3), name="roLimbVelocitiesEnemy")(_olve)

        _oore = layers.Dense(16, name="oGroinRotationsEnemy")(dense_a)
        ogre = layers.Reshape((4, 4), name="roGroinRotationsEnemy")(_oore)

        _ojse = layers.Dense(22, name="oJointStatesEnemy")(dense_a)
        ojse = layers.Reshape((1, 22), name="roJointStatesEnemy")(_ojse)

        __olpe = layers.Dense(dbrain, activation="swish")(_olpe)
        __olve = layers.Dense(dbrain, activation="swish")(_olve)
        __oore = layers.Dense(dbrain, activation="swish")(_oore)
        __ojse = layers.Dense(dbrain, activation="swish")(_ojse)

        __olpe = layers.Conv1DTranspose(filters=15, kernel_size=16, padding="same", data_format="channels_first")(__olpe)
        __olve = layers.Conv1DTranspose(filters=15, kernel_size=16, padding="same", data_format="channels_first")(__olve)
        __oore = layers.Conv1DTranspose(filters=15, kernel_size=16, padding="same", data_format="channels_first")(__oore)
        __ojse = layers.Conv1DTranspose(filters=15, kernel_size=16, padding="same", data_format="channels_first")(__ojse)

        __olve = layers.Reshape((1, 15, dbrain))(__olve)
        __oore = layers.Reshape((1, 15, dbrain))(__oore)
        __ojse = layers.Reshape((1, 15, dbrain))(__ojse)

        __olpe = layers.Reshape((1, 15, dbrain))(__olpe)
        __olve = layers.Reshape((1, 15, dbrain))(__olve)
        __oore = layers.Reshape((1, 15, dbrain))(__oore)
        __ojse = layers.Reshape((1, 15, dbrain))(__ojse)

        sstlp_e = layers.Dense(dbrain, activation="swish")(stlp_e)
        sstlv_e = layers.Dense(dbrain, activation="swish")(stlv_e)
        sstgr_e = layers.Dense(dbrain, activation="swish")(stgr_e)
        sstjs_e = layers.Dense(dbrain, activation="swish")(stjs_e)

        sstlp_e = layers.Reshape((1, 15, dbrain))(sstlp_e)
        sstlv_e = layers.Reshape((1, 15, dbrain))(sstlv_e)
        sstgr_e = layers.Reshape((1, 15, dbrain))(sstgr_e)
        sstjs_e = layers.Reshape((1, 15, dbrain))(sstjs_e)

        sstlp_y = layers.Dense(dbrain, activation="swish")(stlp_y)
        sstlv_y = layers.Dense(dbrain, activation="swish")(stlv_y)
        sstgr_y = layers.Dense(dbrain, activation="swish")(stgr_y)
        sstjs_y = layers.Dense(dbrain, activation="swish")(stjs_y)

        sstlp_y = layers.Reshape((1, 15, dbrain))(sstlp_y)
        sstlv_y = layers.Reshape((1, 15, dbrain))(sstlv_y)
        sstgr_y = layers.Reshape((1, 15, dbrain))(sstgr_y)
        sstjs_y = layers.Reshape((1, 15, dbrain))(sstjs_y)

        concat_b = layers.concatenate([
            sstlp_e,
            sstlv_e,
            sstgr_e,
            sstjs_e,
            sstlp_y,
            sstlv_y,
            sstgr_y,
            sstjs_y,
            __olpe,
            __olve,
            __oore,
            __ojse
        ], axis=1)
        x = layers.ConvLSTM1D(
            filters=512, kernel_size=15, activation="swish",
            return_sequences=False, dropout=0.01, recurrent_dropout=0.01
        )(concat_b)

        _olpy = layers.Dense(63, name="oLimbPositionsYou")(x)
        olpy = layers.Reshape((21, 3), name="roLimbPositionsYou")(_olpy)
        _olvy = layers.Dense(63, name="oLimbVelocitiesYou")(x)
        olvy = layers.Reshape((21, 3), name="roLimbVelocitiesYou")(_olvy)
        _oory = layers.Dense(16, name="oGroinRotationsYou")(x)
        ogry = layers.Reshape((4, 4), name="roGroinRotationsYou")(_oory)
        _ojsy = layers.Dense(22, name="oJointStatesYou")(x)
        ojsy = layers.Reshape((1, 22), name="roJointStatesYou")(_ojsy)

        model = keras.Model(
            inputs=[stlp_y, stlp_e, stlv_y, stlv_e, stgr_y, stgr_e, stjs_y, stjs_e],
            outputs={
                "roLimbPositionsYou": olpy,
                "roLimbPositionsEnemy": olpe,
                "roLimbVelocitiesYou": olvy,
                "roLimbVelocitiesEnemy": olve,
                "roGroinRotationsYou": ogry,
                "roGroinRotationsEnemy": ogre,
                "roJointStatesYou": ojsy,
                "roJointStatesEnemy": ojse
            },
        )
        model.compile(
            optimizer=keras.optimizers.Adadelta(0.01),
            loss={
                "roLimbPositionsYou": keras.losses.MeanAbsoluteError(),
                "roLimbPositionsEnemy": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesYou": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesEnemy": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsYou": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsEnemy": keras.losses.MeanAbsoluteError(),
                "roJointStatesYou": keras.losses.MeanAbsoluteError(),
                "roJointStatesEnemy": keras.losses.MeanAbsoluteError(),
            }
        )
        model.summary()
        self._visualize_model(model, name="modelAdvanced2.png")
        model.load_weights("./Datav4/Model.ckpt")
        return model
