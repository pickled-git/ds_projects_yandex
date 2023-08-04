## [EN] Determining Customer Age

### Project Objective

To build a model that will determine an approximate age of a person from a photograph.

### Project Description

We have a dataset with people's photographs from [ChaLearn Looking at People](http://chalearnlap.cvc.uab.es/dataset/26/description/), containing 7591 photographs. We already have preprocessed photos of faces - there's no need to perform face detection.

The training code was executed on a GPU on a separate server. There is no direct access to the server kernel - it imports and executes functions in a specified format:

- `load_train(path)` - loads the training dataset,
- `load_test(path)` - loads the test dataset,
- `create_model(input_shape)` - creates network architecture, compiles the model,
- `train_model(model, train_data, test_data, batch_size, epochs, steps_per_epoch, validation_steps)` - trains the model.
  
The goal is to achieve a MAE value on the test set not exceeding 8.

## [RUS] Определение возраста покупателей

### Задача проекта

Построить модель, которая по фотографии определит приблизительный возраст человека.

### Описание проекта

Имеется датасет с фотографиями людей [ChaLearn Looking at People](http://chalearnlap.cvc.uab.es/dataset/26/description/), содержащий 7591 фотографию. В нашем распоряжении уже предобработанные фотографии лиц - задачу детекции выполнять не нужно.

Код обучения выполнялся на GPU на отдельном сервере. Прямого доступа к ядру сервера нет - он импортирует и выполняет функции в заданном формате: 

- `load_train(path)` - загрузка тренировочного датасета,
- `load_test(path)` - загрузка тестового датасета,
- `create_model(input_shape)` - создание архитектуры сети, компиляция модели,
- `train_model(model, train_data, test_data, batch_size, epochs, steps_per_epoch, validation_steps)` - обучение модели.

Необходимо добиться значения MAE на тестовой выборке не больше 8.
