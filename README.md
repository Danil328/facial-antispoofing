# Facial-antispoofing

Спуфинг в биометрии (spoofing, атака с подменой личности) - ахиллесова пята многих биометрических систем идентификации и верификации. Подобным атакам подвержены практически все используемые модальности, но в первую очередь - голос и лицо. Спуфинг в системах лицевой биометрии, работающих с RGB-изображениями или видеорядом, может происходить по сценарию атаки повторного воспроизведения (video replay-attack): осуществляется съемка экрана, на котором проигрывается видеофайл с лицом взламываемой персоны. Эффект данного вид атаки усиливается, если биометрическая система работает в режиме frictionless, когда верификация пользователя выполняется параллельно его взаимодействию с приложением, а не прерывает его.

## Выборка
База IDRND_FASDB - это кадры из видеозаписей, снятых в различных условиях. 
Оригинальные видеозаписи получены с помощью веб-камер, камер мобильных телефонов или скачаны c Youtube.
Атаки повторного воспроизведения получены путем съемки экранов различных ноутбуков и мониторов (120 шт) камерами мобильных телефонов (111 шт) в момент проигравания оригинальных видеозаписей различных персон (115 чел), полученных с помощью различных устройств видеосъемки (12 шт).
Атаки были получены как в лабораторных условиях, так и с помощью исполнителей, зарегистрированных в краудсорсинговых интернет-сервисах Яндекс.Толока и Amazon Mechanical Turk.

## Особенности jupyter notebook в папке train
Все вычисления, за исключением финального предсказания, производились на https://colab.research.google.com/
поэтому, в ноубуках сперва производится загрузка данных.

## Model weights
    https://drive.google.com/uc?id=15TKOlGEgk-3m6R8TeMzFxn_RQzwB6hjM
    https://drive.google.com/uc?id=1bhLypjl0BNtaFSjOLlICO0ecCkZ-rNKF
    https://drive.google.com/uc?id=1hrkhV4JdN_JpHkkxpGnBVGeBzKLnBRMz
    https://drive.google.com/uc?id=1Giv-wG23AqZ9IEnwfDUJlHULPn7Tsb8C
    https://drive.google.com/uc?id=1-82xRiQCGyzIeO4krbPtxvRde-s4fo-b


Имена файлов из _train:

**real**: `real/<source code>_id<user id>_*.png` <br/>
**spoof**: `spoof/<source code>_<screen code>_<device code>_id<user id>_*.png`

## Docker
Сборка образа для докер контейнера производится следующей командой:

    docker build -t danil328/antispoofing .
    
Или:

    docker pull danil328/antispoofing

Создание контейнера и запуск:

    nvidia-docker run -v 'test_dir':/test -v 'output_dir':/output -it danil328/antispoofing /bin/bash
    nvidia-docker run --mount src="/test",target=/test,type=bind --mount src="/output",target=/output,type=bind -it danil328/antispoofing /bin/bash 

    
Запуск predict:

    cd root
    python my_main.py


**Точки монтирования /test и /output - обязательны для оценки решений. Формат выходного решения представлен в примере.**
