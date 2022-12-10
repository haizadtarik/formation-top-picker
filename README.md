# formation-top-picker

## Problem
Interpreting formation tops is a task undertaken by a geoscientist to
determine significant boundaries in the subsurface to identify correlation
between the wells or to develop a geological concept of the area. However, 
there are many wells need to be interpreted during the development
phase which will be time consuming to go through all of them.


## Solution
This repo is a ML based framework and app to train a machine learning model and
perform prediction for picking formation tops or lithology zoning using the well 
logs provided by user. It also allow user to visualize the prediction through 
multiple colours of lithology zoning for easy interpretation by geologist.

 ## Quick Start

 1. Install using pip
    ```
    pip install git+https://github.com/haizadtarik/formation-top-picker
    ```

## Run WebApp

1. Run the following command to launch the web apps:
    ```
    streamlit run app.py
    ```

## Run in your code

1. Load and process the data 
    ```
    from toppicker import WellData

    data = WellData(<PATH_TO_LAS_FILES_DIRECTORY>,[<FEATURE_NAME>])
    train_df = data.process_data()
    ```

2. Train a new model and save
    ```
    from toppicker import Trainer
    
    trainer = Trainer()
    trainer.train(<TRAIN_DATAFRAME>)
    trainer.save(<FILENAME>)
    ```

3. Load trained model and perform prediction
    ```
    from toppicker import Trainer
    
    trainer = Trainer()
    trainer.load(<FILENAME>)
    yhat = trainer.predict(<TEST_DATAFRAME>)
    ```

## References

1. McDonald, A., 2021, Python and Petrophysics Notebook Series. https://github.com/andymcdgeo/Petrophysics-Python-Series