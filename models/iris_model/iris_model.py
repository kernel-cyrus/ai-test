'''
    Run on Orin (Jatpack 6.1)
    -------------------------------
    1. Install Dependency
    pip install pandas scikit-learn tf2onnx
    
    2. Install Tensorflow (NV Special Release)
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v61 tensorflow==2.16.1+nv24.8

    3. Get GPU Loading
    watch -n 1 cat /sys/devices/platform/gpu.0/load
'''

import os
import tensorflow as tf
import pandas as pd
import sklearn as sk
import numpy as np

def train_model():

    dataset_path = './dataset/iris.data'

    # Read dataset
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv(dataset_path, names=columns)
    # print(df)

    # Encode dataset (encode species to id)
    species_encoder = sk.preprocessing.LabelEncoder()
    df['class_id'] = species_encoder.fit_transform(df['species'])

    # Prepare train dataset and test dataset
    df_x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    df_y = df['class_id']
    train_x, test_x, train_y, test_y = sk.model_selection.train_test_split(df_x, df_y, test_size=0.2, random_state=42)

    # Randomized train dataset
    # print(train_x)
    # print(train_y)

    # Randomized test dataset
    # print(test_x)
    # print(test_y)

    # Create tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_x.values, train_y.values)).batch(16)
    testset = tf.data.Dataset.from_tensor_slices((test_x.values, test_y.values)).batch(16)
    #print(list(dataset))
    #print(list(testset))

    # Create 3-layers model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model (epochs is training round)
    model.fit(dataset, epochs=500)

    # Test
    loss, accurate = model.evaluate(testset)
    print('Accurate: %.4f' % accurate)

    return model

if __name__ == '__main__':

    # Disable CUDA
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Show device info
    print("GPU Detect:", tf.config.list_physical_devices('GPU'))
    print("Using Device:", tf.test.gpu_device_name())

    saved_path = './saved_model'
    keras_path = './iris_model.keras'
    onnx_path  = './iris_model.onnx'

    if os.path.exists(keras_path):

        # Load Model
        print('Load model:', keras_path)
        model = tf.keras.models.load_model(keras_path)

    else:

        # Train Model
        print('Train model...')
        model = train_model()

        # Save Model
        model.save(keras_path)
        print('Model saved:', keras_path)
        
        # Convert to ONNX
        print('Generating ONNX...')
        model.export(saved_path)
        os.system('python3 -m tf2onnx.convert --saved-model ./saved_model --output ./iris_model.onnx --opset 13')

    # Print model summary
    model.summary()

    # Prediction
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    sample = np.array([[5.1, 3.5, 1.4, 0.2]]) # Iris-setosa

    pred_probs = model.predict(sample)
    class_id = np.argmax(pred_probs, axis=1)[0]

    # Print Result
    print("Sample:", sample)
    print("Prediction:", species, pred_probs)
    print("Result:", species[class_id])