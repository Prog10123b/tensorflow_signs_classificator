from image_classification import classificator
import numpy as np

classific = classificator()

classific.load_model('save\\save_00')

predictions = classific.model.predict(image) # image format is np.array([img, ])
print(np.argmax(predictions[0])) # returns number of class
# classes: ['stop', 'only-right', 'only-left', 'only-straight', 'straight-right', 'straight-left']
