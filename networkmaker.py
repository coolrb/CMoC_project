from keras.models import Model
from keras.layers import Input, LSTM

oneHotDim = 37

inputLayer = Input(shape=(None,oneHotDim))

model = LSTM(50, activation="relu",return_sequences=True)(inputLayer)
model = LSTM(50, activation="relu",return_sequences=True)(model)
model = LSTM(1, activation="sigmoid",return_sequences=True)(model)

model = Model(inputLayer,model)
