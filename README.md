# Распознавание рукописных символов EMNIST с помощью модели CNN

## 1. Описание решения
- _тип задачи_ - создание веб-приложения для распознавания самостоятельно нарисованных символов с помощью свёрточной нейронной сети
- данные представляют собой изображения латинских букв и цифр размером 28x28 px, как в данных примерах:\
![Image1](https://github.com/Wizardass/cnn-image-classification/blob/main/sample_images/image_0_label_45.png)
![Image2](https://github.com/Wizardass/cnn-image-classification/blob/main/sample_images/image_1_label_36.png)
![Image3](https://github.com/Wizardass/cnn-image-classification/blob/main/sample_images/image_2_label_43.png)
![Image4](https://github.com/Wizardass/cnn-image-classification/blob/main/sample_images/image_3_label_15.png)
![Image5](https://github.com/Wizardass/cnn-image-classification/blob/main/sample_images/image_4_label_4.png)


- *Что представляет собой класс CNN:*
  - **Конструктор `__init__`**:
  - Принимает параметр `n_classes` (количество классов для классификации).
  - Определяет 3 компонента сети: `encoder`, `decoder` и `classifier`.

  - **Компоненты сети**:
    - *Encoder*: Последовательность свёрточных слоев, включающих в себя:
      - Свёрточный слой (`nn.Conv2d`)
      - Max Pooling слой (`nn.MaxPool2d`)
      - Слой нормализации (`nn.BatchNorm2d`)
      - Функция активации (`nn.ReLU`)
      - Слой для регуляризации (`nn.Dropout`)
    - *Decoder*: Последовательность слоев для декодирования, включающих в себя:
      - Обратный свёрточный слой (`nn.ConvTranspose2d`)
      - Слой нормализации (`nn.BatchNorm2d`)
      - Функция активации (`nn.ReLU`)
      - Слой для регуляризации (`nn.Dropout`)
    - *Classifier*: Последовательность полносвязных слоев для классификации, включающая в себя:
      - Слой для преобразования тензора в вектор (`nn.Flatten`)
      - Три линейных слоя (`nn.Linear`) с функцией активации ReLU и Dropout для регуляризации

  - **Метод `forward`**:
    - Определяет проход через сеть:
    1. Входные данные проходят через `encoder`.
    2. Далее через `decoder`.
    3. Затем через `classifier`.
    - Возвращает выходные данные сети.

- accuracy score модели составил 90.2%


## 2. Установка и запуск сервиса
Для быстрого создания веб-приложения через Docker требуется предварительно скачать предобученную модель (model.ckpt) по адресу:
https://drive.google.com/file/d/1nh81UlhjbzxFoVGNG5CgokufKIs8nGYl/view?usp=sharing

```bash
git clone https://github.com/Wizardass/cnn-image-classification.git
```

Размещаем ранее скачанный файл model.ckpt в папке myapp.

```bash
cd cnn-image-classification/
docker build -t predict_symbol .
docker run -p 8000:8000 predict_symbol
```

Затем нужно открыть браузер, перейти по адресу 127.0.0.1:8000, нарисовать интересующий символ для распознавания и нажать predict. Чтобы стереть символ и нарисовать другой, нужно нажать clear.