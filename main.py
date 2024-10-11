import cv2 as cv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling3D, Dropout, Flatten, Dense, Conv3D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

img_width = 64
img_height = 64
sample_period = 32

class ViolenceDetectionModel:
    def __init__(self, img_width, img_height, sample_period):
        self.img_width = img_width
        self.img_height = img_height
        self.sample_period = sample_period
        self.model = self.build_model()
        self.results = []

    def video_extractor(self, video_path, num_samples):
        cap = cv.VideoCapture(video_path)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        sample_step = max(frame_count // num_samples, 1)
        frames = []
        for i in range(num_samples):
            cap.set(cv.CAP_PROP_POS_FRAMES, i * sample_step)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (self.img_width, self.img_height))
            frame = frame / 255
            frames.append(frame)
        cap.release()
        return frames

    def create_dataset(self, base_path, num_samples):
        frame_list = []
        label_list = []
        for class_id, class_name in enumerate(os.listdir(base_path)):
            class_path = os.path.join(base_path, class_name)
            for video_name in os.listdir(class_path):
                video_path = os.path.join(class_path, video_name)
                frames = self.video_extractor(video_path, num_samples)
                frame_list.append(frames)
                label_list.append(class_id)
        return np.array(frame_list), np.array(label_list)

    def build_model(self):
        model = Sequential()
        model.add(Conv3D(4, (3, 3, 3), activation='relu', input_shape=(self.sample_period, self.img_height, self.img_width, 3), padding='same'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv3D(8, (3, 3, 3), activation='relu', padding='same'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv3D(14, (3, 3, 3), activation='relu', padding='same'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, frame_list, label_list, epochs=50, batch_size=32):
        x_train, x_test, y_train, y_test = train_test_split(frame_list, label_list, test_size=0.2)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[early_stopping])
        return history

    def detect_violence(self, video_path):
        cap = cv.VideoCapture(video_path)
        frame_list = []
        statu = False
        fps_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            fps_counter += 1
            if statu:
                cv.rectangle(frame, (0, 0), (300, 50), (255, 0, 0), -1)
                cv.putText(frame, "Violence Detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            else:
                cv.rectangle(frame, (0, 0), (350, 50), (255, 0, 0), -1)
                cv.putText(frame, "No Violence Detected", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            crop_frame = cv.resize(frame, (self.img_width, self.img_height))
            if len(frame_list) < self.sample_period:
                frame_list.append(crop_frame)

            if len(frame_list) == self.sample_period and fps_counter % 10 == 0:
                frame_list_array = np.array(frame_list)
                result = self.model.predict(frame_list_array.reshape((1, self.sample_period, self.img_width, self.img_height, 3)), verbose=0)
                self.results.append(result[0][0])
                statu = result[0][0] >= 0.5

                frame_list.append(crop_frame)
                frame_list.pop(0)

            cv.imshow('Violence Detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def plot_results(self):
        plt.plot(self.results, label="Violence Detection Score")
        plt.xlabel('Frames')
        plt.ylabel('Detection Score')
        plt.title('Violence Detection Over Time')
        plt.legend()
        plt.show()



base_path = r"C:\Users\tarik\PycharmProjects\pythonProject\violance_detection\Real Life Violence Dataset"
model = ViolenceDetectionModel(img_width, img_height, sample_period)

frame_list, label_list = model.create_dataset(base_path, 40)
model.train(frame_list, label_list)

test_video_path = r"C:\Users\tarik\PycharmProjects\pythonProject\violance_detection\punisher_video.mp4"
model.detect_violence(test_video_path)

