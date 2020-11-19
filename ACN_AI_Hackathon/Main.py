from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
import os
from ResNet import create_res_net

def main(): 
    try: 
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        model = create_res_net()
        model.summary()

        print ("size(X_TRAIN): ", size(x_train))
        print ("size(Y_TRAIN): ", size(y_train))

        timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = 'cifar-10_res_net_30-'+timestr

        checkpoint_path = "checkpoints/"+name+"/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.system('mkdir {}'.format(checkpoint_dir))

        # save model after each epoch
        cp_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1
        )
        tensorboard_callback = TensorBoard(
            log_dir='tensorboard_logs/'+name,
            histogram_freq=1
        )

        model.fit(
            x=x_train,
            y=y_train,
            epochs=20,
            verbose=1,
            validation_data=(x_test, y_test),
            batch_size=128,
            callbacks=[cp_callback, tensorboard_callback]
        )
    except IOError as ex:
        print(ex)
        pass

if __name__ == "__main__": 
    main()

