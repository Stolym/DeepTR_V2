from datetime import datetime

import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers

gpu = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpu[0], True)


def ff64(self):
    return np.array(self, np.float)

class Brain:
    def __init__(self, **kwargs):
        self._model = self._mount_model()
        self.logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.logdir)


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

    def train(self, fbatch):
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
            filepath="./Data/Model.ckpt",
            verbose=1,
            save_weights_only=True,
            save_freq=64
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
            optimizer=keras.optimizers.RMSprop(1e-5),
            loss={
                "roLimbPositionsYou": keras.losses.MeanAbsoluteError(),
                "roLimbPositionsEnemy": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesYou": keras.losses.MeanAbsoluteError(),
                "roLimbVelocitiesEnemy": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsYou": keras.losses.MeanAbsoluteError(),
                "roGroinRotationsEnemy": keras.losses.MeanAbsoluteError(),
                "roPlayerSelection": keras.losses.MeanAbsoluteError(),
                "roInjuries": keras.losses.MeanAbsoluteError(),
                "roJointStatesYou": keras.losses.MeanAbsoluteError(),
                "roJointStatesEnemy": keras.losses.MeanAbsoluteError(),
            }
        )
        model.summary()
        self._visualize_model(model, name="modelStandard.png")
        model.load_weights("./Data/Model.ckpt")
        return model
