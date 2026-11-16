% Disease Facts
disease(diabetes).
disease(obesity).
disease(high_bp).
disease(anemia).
disease(heart_disease).

% Diet Recommendations
diet(diabetes, low_sugar).
diet(diabetes, whole_grains).

diet(obesity, low_calorie).
diet(obesity, fruits_and_vegetables).

diet(high_bp, low_salt).
diet(high_bp, fruits_and_vegetables).

diet(anemia, iron_rich_food).
diet(anemia, green_leafy_vegetables).

diet(heart_disease, low_fat).
diet(heart_disease, high_fiber).

% Rule
suggest_diet(Disease, Diet) :-
    disease(Disease),
    diet(Disease, Diet).