# Цветочный классификатор [RU]
#
## Описание проекта
Этот проект представляет собой реализацию нейронной сети для классификации изображений цветов с использованием библиотеки PyTorch. Проект включает в себя несколько этапов: подготовку данных, создание нейронной сети, обучение модели и оценку ее производительности.

## Подготовка данных
В проекте используется набор данных цветов (flowers dataset), который должен быть загружен и распакован в папку /content/flowers. В этой папке должны быть подпапки с изображениями для каждого класса цветка.

Данные подготавливаются с использованием библиотеки torchvision и преобразований. Изображения масштабируются, конвертируются в тензоры и нормализуются.

Датасет разделяется на тренировочную и тестовую выборки в пропорции 85% к 15%.

## Создание нейронной сети
Для решения задачи классификации изображений была создана сверточная нейронная сеть (CNN) с несколькими сверточными и полносвязанными слоями.

## Архитектура нейронной сети:

Первый сверточный слой с 16 фильтрами, размером ядра 11x11 и шагом 4.
Дополнительные сверточные слои, слои пулинга и полносвязные слои.
Функции активации ReLU используются между слоями.
Модель выводит вероятности принадлежности изображения к одному из 5 классов цветов.

## Обучение модели
Модель обучается на тренировочной выборке с использованием функции потерь CrossEntropyLoss и оптимизатора Adam.

Обучение происходит в течение нескольких эпох (15 в данном случае).

После каждой эпохи выводятся потери (loss) и график изменения потерь во времени.

## Оценка производительности
После обучения модели оценивается ее производительность на тестовой выборке.

Выводится точность классификации для каждого класса цветка.

Модель сохраняется в файл для будущего использования.

## Запуск проекта
Загрузите код из репозитория.

Убедитесь, что данные находятся в правильном месте (папка /content/flowers).

Запустите код в среде с GPU-поддержкой, если доступно, для ускорения обучения.

## Зависимости
Проект требует следующие зависимости:

PyTorch
torchvision
tqdm (для отображения прогресса обучения)
## Замечания
Этот проект является примером обучения нейронной сети для классификации изображений и может быть использован в качестве отправной точки для более сложных задач.
#
#
#
#
#

# FlowerClassifier [ENG]
#
## Project Description
This project is an implementation of a neural network for image classification of flowers using the PyTorch library. The project consists of several stages: data preparation, neural network creation, model training, and performance evaluation.

## Data Preparation
The project uses the flowers dataset, which should be downloaded and unpacked into the /content/flowers directory. Inside this folder, there should be subfolders with images for each flower class.

Data is prepared using the torchvision library and transformations. Images are resized, converted to tensors, and normalized.

The dataset is split into a training and testing set in an 85% to 15% ratio.

## Creating a Neural Network
To solve the image classification task, a convolutional neural network (CNN) with multiple convolutional and fully connected layers was created.

## Network architecture:

The first convolutional layer with 16 filters, a kernel size of 11x11, and a stride of 4.
Additional convolutional layers, pooling layers, and fully connected layers.
ReLU activation functions are used between layers.
The model outputs probabilities of images belonging to one of the 5 flower classes.

## Training the Model
The model is trained on the training dataset using the CrossEntropyLoss loss function and the Adam optimizer.

Training takes place over several epochs (15 in this case).

After each epoch, losses are displayed, and a graph of loss changes over time is plotted.

## Performance Evaluation
After model training, its performance is evaluated on the testing dataset.

Classification accuracy is displayed for each flower class.

The model is saved to a file for future use.

## Running the Project
Download the code from the repository.

Ensure that the data is in the correct location (the /content/flowers folder).

Run the code in an environment with GPU support, if available, to speed up training.

## Dependencies
The project requires the following dependencies:

PyTorch
torchvision
tqdm (for displaying training progress)
## Notes
This project serves as an example of training a neural network for image classification and can be used as a starting point for more complex tasks.
