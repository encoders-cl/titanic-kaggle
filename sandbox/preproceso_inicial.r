library(data.table)
library(tidyverse)
library(ranger)
library(h2o)

# Lectura de datos --------------------------------------------------------

data_gender = fread("files/datasets/input/gender_submission.csv")
data_test = fread("files/datasets/input/test.csv")
data_master = fread("files/datasets/input/train.csv")

# Splitear ----------------------------------------------------------------
set.seed(1234)
sampleo = sample(1:nrow(data_master), 0.7 * nrow(data_master))
data_train = data_master[sampleo,]
data_valid = data_master[-sampleo,]

# Hacer modelo ------------------------------------------------------------

data_train$Survived  = as.factor(data_train$Survived)
data_train$Age[is.na(data_train$Age)] = mean(data_train$Age, na.rm = T)
data_valid$Age[is.na(data_valid$Age)] = mean(data_valid$Age, na.rm = T)

data_test$Age[is.na(data_test$Age)] = mean(data_test$Age, na.rm = T)
data_test$Fare[is.na(data_test$Fare)] = mean(data_test$Fare, na.rm = T)

modelRF <- ranger(Survived ~ ., data = data_train, num.trees =  100)


# Hacer modelo a la bruta H2O ----------------------------------------

h2o.init()
train_h20 <- as.h2o(data_train, destination_frame = "tablon_train")
test_h20 <- as.h2o(data_test, destination_frame = "tablon_test")
valid_h20 <- as.h2o(data_valid, destination_frame = "tablon_valid")


respuesta = 'Survived'
atributos = setdiff(colnames(data_master),c(respuesta,'Survived'))
atributos

modelo_h2o = h2o.automl(y = respuesta,
                    x = atributos,
                    training_frame =  train_h20,
                    validation_frame = valid_h20,
                    max_runtime_secs = 10*60,
                    balance_classes = F
)

modelo_h20_leader <- modelo_h2o@leader

performance_valid <- h2o.performance(modelo_h20_leader, newdata = valid_h20)
h2o.confusionMatrix(performance_valid)
h2o.auc(performance_valid)

# Sacar score -------------------------------------------------------------

prediccion_en_validacion = predict(modelo_h20_leader, valid_h20)

data_valid$Survived
prediccion_en_validacion$predictions

table(prediccion_en_validacion$predictions, data_valid$Survived)
accuracy = (145 + 76)/(145 + 22 +25 +76)


# Sacar Score de Test ------------------------------------------------------

prediccion_en_test = predict(modelo_h20_leader, test_h20)

prediccion_en_test = h2o.predict(modelo_h20_leader, test_h20)

# Guardar el archivo ------------------------------------------

vector_de_predicciones = prediccion_en_test$predict |> as.data.frame() |> unlist() |> unname()

data_para_submitear = 
data.frame(PassengerId = data_test$PassengerId, 
Survived = vector_de_predicciones) |> 
fwrite("files/datasets/output/data_en_coders_kaggle_titanic.csv")



#----h20 -----------------
