"""
==============================================================================
FASE 3: Selección de biomarcadores y reducción de dimensionalidad
==============================================================================
Archivo: 03_Seleccion_Variables.py
Este script ejecuta el pipeline de reducción de dimensionalidad para aislar el
"Set clínico óptimo" de biomarcadores. Al ser un proceso de alto coste
computacional, se ha encapsulado en un script puro de Python para maximizar
la eficiencia de ejecución.

Metodología:
1. Filtro de colinealidad: Eliminación de redundancia matemática (Pearson > 0.85).
   Vital para proteger la interpretabilidad en fases posteriores.
2. Eliminación recursiva (RFECV): Se utiliza Random Forest como estimador base.
   * Criterio de optimización: Se optimiza estrictamente la métrica 'Recall' 
     (Sensibilidad) para blindar el modelo contra Falsos Negativos (el error 
     más crítico en medicina preventiva).
==============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import joblib
import time

def main():

    # 1. Carga de datos de la fase 2

    try:
        datos_fase2 = joblib.load('../models/datos_preprocesados_fase2.pkl')
    except FileNotFoundError:
        print("No se encuentra 'datos_preprocesados_fase2.pkl'")
        return

    # Extraemos los datos listos para usar
    X_train_scaled = datos_fase2['X_train']
    X_test_scaled = datos_fase2['X_test']
    y_train = datos_fase2['y_train']
    y_test = datos_fase2['y_test']
    
    print(f"Pacientes en entrenamiento: {X_train_scaled.shape[0]}")
    print(f"Variables clínicas iniciales: {X_train_scaled.shape[1]}")
    

    # 2. filtro de colinealidad: eliminamos variables redundantes (Pearson > 0.85)
    corr_matrix = X_train_scaled.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Encontramos variables redundantes
    variables_a_eliminar = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]
    
    # Filtramos solo en Train por ahora
    X_train_filtrado = X_train_scaled.drop(columns=variables_a_eliminar)
    
    print(f"Variables eliminadas por ser matemáticamente redundantes: {len(variables_a_eliminar)}")
    print(f"Variables restantes para el análisis RFE: {X_train_filtrado.shape[1]}")
    
    # 3. RFE: Eliminación recursiva para encontrar el set mínimo clínico

    start_time = time.time()
    evaluador = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    cv_clinico = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Optimizamos por Recall (Sensibilidad) para no dejar escapar falsos negativos
    selector_rfe = RFECV(
        estimator=evaluador, 
        step=2,               
        cv=cv_clinico, 
        scoring='recall',     
        n_jobs=-1             
    )
    
    selector_rfe = selector_rfe.fit(X_train_filtrado, y_train)
    end_time = time.time()
    
    # 4.Preparamos los datos finales para la siguiente fase de modelado

    biomarcadores_finales = X_train_filtrado.columns[selector_rfe.support_].tolist()
    
    # Recortamos Train y Test para que solo tengan los biomarcadores ganadores definitivos
    X_train_final = X_train_filtrado[biomarcadores_finales]
    X_test_final = X_test_scaled[biomarcadores_finales]
    
    print(f"Proceso completado en {round((end_time - start_time)/60, 2)} minutos")
    print(f"De las 450 variables originales, el modelo ha aislado un 'Set Clínico Óptimo'")
    print(f"compuesto por: {selector_rfe.n_features_} biomarcadores críticos")
    
    # Guardamos el dataset totalmente limpio, escalado y filtrado
    datos_modelado = {
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train,
        'y_test': y_test,
        'features': biomarcadores_finales
    }
    
    joblib.dump(datos_modelado, '../models/dataset_optimizado_fase3.pkl')

if __name__ == "__main__":
    main()