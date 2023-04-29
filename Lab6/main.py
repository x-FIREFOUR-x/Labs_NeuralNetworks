import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.callbacks import ModelCheckpoint

from model import Xception
from processingData.preprocessingData import PreprocessingData
from processingData.dataset import Dataset

from arguments import DATASET_PATH, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_NAMES, PATH_SAVE_MODEL, EPOCHS


def train_model(train_ds, validation_ds, test_ds, epochs, batch_size):
    model = Xception((IMAGE_HEIGHT, IMAGE_WIDTH, 3), len(CLASSES_NAMES)-1)

    initial_learning_rate = 10 ** (-3)
    final_learning_rate = 10 ** (-7)
    learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / epochs)
    steps_per_epoch = len(train_ds)

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch,
        decay_rate=learning_rate_decay_factor
    )

    model.compile(loss='binary_crossentropy', #loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate))

    model.summary()

    checkpoint_dir = PATH_SAVE_MODEL + "/Checkpoints/"
    checkpoint_path = checkpoint_dir + "cp-{epoch:04d}.ckpt"
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=True,
                                 mode='auto')

    tf_path = PATH_SAVE_MODEL + "/Model/"
    fullModelSave = ModelCheckpoint(filepath=tf_path,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='auto')

    log_dir = PATH_SAVE_MODEL + "/Logs/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    callbacks_list = [checkpoint, tensorboard_callback, fullModelSave]

    model.fit(
        train_ds,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_ds,
        callbacks=callbacks_list,
        verbose=1)

    return model

def predict_image(model, test_data, count):
    plt.figure(figsize=(4, 4))
    for (images,  correct_classes) in test_data.take(count):
        correct_classes = correct_classes.numpy()
        predictions = model.predict(images)[1]
        predicted_classes = np.argmax(predictions)

        for index, img in enumerate(images):
            img = img.numpy()
            print('Correct =', CLASSES_NAMES[correct_classes[index]])
            print('Predicted =', CLASSES_NAMES[predicted_classes])
            plt.imshow(images[index].numpy().astype('uint8'))
            plt.show()
            break


def recognize_image(model, preprocessingData, path):
    img = preprocessingData.process_single_img(path)
    predictions = model.predict(tf.expand_dims(img, axis = 0))[0]
    classes = np.argmax(predictions).squeeze()
    print(CLASSES_NAMES[classes])
    plt.imshow(img.numpy().astype('uint8'))
    plt.show()

def recognize_video(model, preprocessingData, path):
    video = cv2.VideoCapture(path)
    frame_no = 0
    last_predictions = []
    last_result = False
    begin_time = []
    end_time = []
    while video.isOpened():
        ret, frame = video.read()
        if (ret):
            frame_no += 1
            if (frame_no % 4 != 0):
                continue

            process_frame = preprocessingData.preprocess_opencv_img(frame)
            process_frame = np.expand_dims(process_frame, axis=0)
            predictions = model.predict(process_frame, verbose=0)
            pred_class = predictions[0]
            last_predictions.append(pred_class)
            if (len(last_predictions) > 3):
                last_predictions.pop(0)
            avg_prediction = sum(last_predictions) / len(last_predictions)
            if (avg_prediction > 0.5 and not last_result):
                last_result = True
                time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                print("Start =", time)
                begin_time.append(time)

            if (avg_prediction < 0.5 and last_result):
                last_result = False
                time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
                print("End =", time)
                begin_time.append(time)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    dataset = Dataset(data_path=DATASET_PATH, batch_size=BATCH_SIZE, val_percent=0.15, test_percent=0.1)

    preprocessingData = PreprocessingData(CLASSES_NAMES, IMAGE_HEIGHT, IMAGE_WIDTH)
    (train_ds, validation_ds, test_ds) = dataset.create_data_pipelines(preprocessingData)

    print(tf.data.experimental.cardinality(train_ds).numpy())
    print(tf.data.experimental.cardinality(validation_ds).numpy())
    print(tf.data.experimental.cardinality(test_ds).numpy())

    model = train_model(train_ds, validation_ds, test_ds, EPOCHS, BATCH_SIZE)
    #model = tf.keras.models.load_model('SaveModel/Model')

    model.evaluate(test_ds)

    predict_image(model, test_ds, 10)

    recognize_image(model, preprocessingData, "Data\\test\\01.jpg")
    recognize_image(model, preprocessingData, "Data\\test\\02.jpg")
    recognize_image(model, preprocessingData, "Data\\test\\03.jpg")
    recognize_image(model, preprocessingData, "Data\\test\\04.jpg")
    recognize_image(model, preprocessingData, "Data\\test\\05.jpg")
    recognize_image(model, preprocessingData, "Data\\test\\06.jpg")
    recognize_image(model, preprocessingData, "Data\\test\\07.jpg")

    recognize_video(model, preprocessingData, "Data\\test\\video.mp4")
