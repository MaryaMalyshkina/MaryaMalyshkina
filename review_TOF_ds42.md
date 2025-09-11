# Ревью проекта "TOF" (DS42)

## Сильные стороны

Отличная обработка пропусков.
Наглядный EDA: missingno, распределения, корреляции (φ_K) — помогает увидеть проблемы данных заранее.
Борьба с мультиколлинеарностью и вариантность данных: сделаны два датафрейма (очищенный и с инженерингом признаков), что позволяет сравнить подходы.
Разнообразие алгоритмов: от логистической регрессии и SVC до ансамблей (RF, LGBM, CatBoost) — хорошее покрытие модельного пространства.


## Недочёты, на которые просим обратить внимание:

Пожелание по поводу графиков категориальных признаков: сделать их размер поменьше. Если для признака "створки_кл_ла", у которого много категорий, размер (10, 7) вполне оправдан, то для остальных признаков графики можно сделать вчетверо меньше, это будет удобнее для восприятия.

---

**Матрица ошибок: в комментарии SVC, в коде — CatBoost (operation_1)**

- **Комментарий:**
  `# выведем матрицу ошибок для лучшей прогностической модели SVC`

- **Код ниже:**
  ```python
  model = search_results['CatBoostClassifier'].best_estimator_
  ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
  plt.title('Матрица ошибок для лучшей модели CatBoostClassifier')
  ```
---

**Важность признаков: строится для CatBoost, заголовок от LGBM**

- **Код (блок "CatBoostClassifier"):**
  ```python
  feature_names = search_results_smote250_2['CatBoostClassifier'].best_estimator_.named_steps[
      'preprocessor'].get_feature_names_out()
  importances = search_results_smote250_2['CatBoostClassifier'].best_estimator_.named_steps[
      'model'].feature_importances_
  ...
  plt.title('Важность признаков для модели LGBMClassifier в порядке убывания')

---

**Оверсэмплинг + автоматический баланс классов у модели одновременно**

- **Используется ресэмплинг в пайплайне:**
  ```python
  final_pipe = ImbPipeline(steps=[
      ('preprocessor', preprocessor),
      ('oversampler', oversampler),   # SMOTE/ADASYN/SMOTETomek/...
      ('model', model)
  ])

И одновременно у моделей включён авто-баланс:

  ```python
  LGBMClassifier(..., class_weight='balanced', ...)
  CatBoostClassifier(..., auto_class_weights='Balanced', ...)
```

Это приводит к «двойному» усилению минорных классов и риску переобучения.
Рекомендуется выбрать одну стратегию балансировки:
либо использовать oversampling, отключив class_weight,
либо оставить только class_weight, но не применять SMOTE/ADASYN.

---
**Опечатка в заголовках секций: `opetration` вместо `operation`**

- **Markdown-заголовки:**
  > `**Для датафрейма *opetration_1***`  
  > `**Для датафрейма *opetration_2***`
---

**Небольшая ремарка по поводу кодирования категориальных признаков.**

Здесь в проекте применен OneHotEncoder для всех моделей, в том числе для Catboost и Lgbm, у которых есть свой встроенный кодировщик.
Эти модели, как известно, специально оптимизированы для категориальных признаков. Если для них использовать OHE,
то они теряют преимущество в обработке категорий. OHE создаёт много новых признаков (по одному на категорию),
что приводит к увеличению размерности. Это замедляет обучение, особенно при большом числе категорий.

В этом проекте  признак "створки_кл_ла" имеет 15 категорий -  не очень много, поэтому вполне допустимо использовать  OneHotEncoder для всех моделей.

Но для лучшего результата можно было бы использовать встроенную обработку для LightGBM и CatBoost.

Модифицированная функция get_preprocessor выглядела бы так:

  ```python
def get_preprocessor(data, model_type='default'):
    X = data.drop(['операция'], axis=1)
    y = data['операция'].copy()
   ...
    # Для CatBoost и LightGBM - особый обработчик категориальных признаков
    if model_type in ['catboost', 'lightgbm']:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
            # Не применяем OneHotEncoder - модели работают с категориями напрямую
        ])
    else:
        # Для остальных моделей используем стандартный подход с OHE
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
...
```
Адаптация функции search_best_models:
```python
def search_best_models(X_train, y_train, X_test, y_test, models_and_params, data):

    search_results = {}
    summary_rows = []

    for name, (model, param_grid) in models_and_params.items():

        # Определяем тип модели для выбора правильного препроцессора
        if name == 'CatBoostClassifier':
            model_type = 'catboost'
        elif name == 'LGBMClassifier':
            model_type = 'lightgbm'
        else:
            model_type = 'default'
        
        # Получаем соответствующий препроцессор
        _, _, _, _, preprocessor = get_preprocessor(data, model_type=model_type)

        # пайплайн с препроцессором и моделью
        final_pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
      ...  
  ```      
Такой подход позволяет CatBoost и LightGBM использовать свои встроенные механизмы обработки категориальных признаков,
что обычно дает лучшие результаты, чем принудительное применение OHE.
  
