# Ревью проекта "TOF" (DS42)

## Сильные стороны

Отличная обработка пропусков.
Наглядный EDA: missingno, распределения, корреляции (φ_K) — помогает увидеть проблемы данных заранее.
Борьба с мультиколлинеарностью и вариантность данных: сделаны два датафрейма (очищенный и с инженерингом признаков), что позволяет сравнить подходы.
Разнообразие алгоритмов: от логистической регрессии и SVC до ансамблей (RF, LGBM, CatBoost) — хорошее покрытие модельного пространства.


## Недочёты, на которые просим обратить внимание:


**Матрица ошибок: в комментарии SVC, в коде — CatBoost (operation_1)**

- **Комментарий:**
  `# выведем матрицу ошибок для лучшей прогностической модели SVC`

- **Код ниже:**
  ```python
  model = search_results['CatBoostClassifier'].best_estimator_
  ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
  plt.title('Матрица ошибок для лучшей модели CatBoostClassifier')

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

Это приводит к «двойному» усилению минорных классов и риску переобучения.
Рекомендуется выбрать одну стратегию балансировки:
либо использовать oversampling, отключив class_weight,
либо оставить только class_weight, но не применять SMOTE/ADASYN.

---

**Опечатка в заголовках секций: `opetration` вместо `operation`**

- **Markdown-заголовки:**
  > `**Для датафрейма *opetration_1***`  
  > `**Для датафрейма *opetration_2***`
