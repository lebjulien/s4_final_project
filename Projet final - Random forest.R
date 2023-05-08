library(readr)
library(randomForest)
library(plotly)

rf_dataset <- clean_data

rf_dataset$Diagnosis <- factor(rf_dataset$Diagnosis)

# Diviser les données en ensembles d'apprentissage et de test
set.seed(123) # pour la reproductibilité des résultats
train_index <- sample(nrow(rf_dataset), 0.7 * nrow(rf_dataset))
train_data <- rf_dataset[train_index, ]
test_data <- rf_dataset[-train_index, ]

# Entraîner le modèle de forêt aléatoire
rf_model <- randomForest(Diagnosis ~ ., data = train_data, ntree = 100, mtry = 2, na.action = na.omit)

# Afficher les résultats du modèle
print(rf_model)

# Calculer l'importance des variables
var_importance <- importance(rf_model)

# Afficher les variables les plus importantes
print(var_importance)

# Évaluer la performance du modèle sur les données de test
rf_predictions <- predict(rf_model, test_data)
confusion_matrix <- table(rf_predictions, test_data$Diagnosis)
accuracy <- sum(diag(confusion_matrix))/sum(confusion_matrix)

# Afficher la matrice de confusion et la précision
print(confusion_matrix)

# Calcul pour la classe M
precision_M <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
recall_M <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
f1_score_M <- 2 * (precision_M * recall_M) / (precision_M + recall_M)

# Calcul pour la classe B
precision_B <- confusion_matrix[1,1] / sum(confusion_matrix[,1])
recall_B <- confusion_matrix[1,1] / sum(confusion_matrix[1,])
f1_score_B <- 2 * (precision_B * recall_B) / (precision_B + recall_B)

# Affichage des résultats
print(paste("F1-score pour la classe Malin :", f1_score_M))
print(paste("F1-score pour la classe Bénin :", f1_score_B))