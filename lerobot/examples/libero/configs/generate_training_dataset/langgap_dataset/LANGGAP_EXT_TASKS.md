# langgap_ext eval suite — task index reference

59 extended tasks (spatial 28 / goal 9 / object 22), suite task_index order below.
`trained` = 학습 데모가 수집된 16개 (LangGap collect 스크립트 task ID 기준, task_registry.py 매핑);
나머지 43개는 held-out (같은 장면, unseen 지시문). `--env.task_ids` 선택 시 이 인덱스 사용.

| idx | trained | task name | instruction |
|---|---|---|---|
| 0 | O | spatial_dim1_change_bowl_ext_01_task0_bowl2 | pick up the black bowl on the right of the ramekin and place it on the plate |
| 1 | O | spatial_dim1_change_bowl_ext_02_task2_bowl2 | pick up the black bowl on the right of the plate and place it on the plate |
| 2 |  | spatial_dim1_change_bowl_ext_03_task3_bowl2 | pick up the black bowl on top of the wooden cabinet and place it on the plate |
| 3 | O | spatial_dim1_change_bowl_ext_04_task4_bowl2 | pick up the black bowl on top of the wooden cabinet and place it on the plate |
| 4 |  | spatial_dim1_change_bowl_ext_05_task8_bowl2 | pick up the black bowl next to the ramekin and place it on the plate |
| 5 | O | spatial_dim2_change_target_ext_01_task0_to_stove | pick up the black bowl between the plate and the ramekin and place it on the stove |
| 6 | O | spatial_dim2_change_target_ext_02_task0_to_cabinet | pick up the black bowl between the plate and the ramekin and place it on the cabinet |
| 7 |  | spatial_dim2_change_target_ext_03_task2_to_stove | pick up the black bowl from the middle of the table and place it on the stove |
| 8 | O | spatial_dim2_change_target_ext_04_task2_to_ramekin | pick up the black bowl from the middle of the table and place it on the ramekin |
| 9 |  | spatial_dim2_change_target_ext_05_task3_to_stove | pick up the black bowl on the cookie box and place it on the stove |
| 10 |  | spatial_dim2_change_target_ext_06_task3_to_cabinet | pick up the black bowl on the cookie box and place it on the cabinet |
| 11 |  | spatial_dim2_change_target_ext_07_task7_to_cabinet | pick up the black bowl on the stove and place it on the cabinet |
| 12 |  | spatial_dim2_change_target_ext_08_task7_to_ramekin | pick up the black bowl on the stove and place it on the ramekin |
| 13 |  | spatial_dim2_change_target_ext_09_task8_to_stove | pick up the black bowl next to the plate and place it on the stove |
| 14 |  | spatial_dim2_change_target_ext_10_task8_to_cabinet | pick up the black bowl next to the plate and place it on the cabinet |
| 15 |  | spatial_dim2_change_target_ext_11_task9_to_stove | pick up the black bowl on the wooden cabinet and place it on the stove |
| 16 |  | spatial_dim2_change_target_ext_12_task9_to_ramekin | pick up the black bowl on the wooden cabinet and place it on the ramekin |
| 17 | O | spatial_dim3_change_object_ext_01_ramekin_to_plate | pick up the ramekin and place it on the plate |
| 18 |  | spatial_dim3_change_object_ext_02_ramekin_to_stove | pick up the ramekin and place it on the stove |
| 19 | O | spatial_dim3_change_object_ext_03_ramekin_to_cabinet | pick up the ramekin and place it on the cabinet |
| 20 |  | spatial_dim3_change_object_ext_04_ramekin_to_cookie_box | pick up the ramekin and place it on the cookie box |
| 21 | O | spatial_dim3_change_object_ext_05_cookie_box_to_plate | pick up the cookie box and place it on the plate |
| 22 |  | spatial_dim3_change_object_ext_06_cookie_box_to_stove | pick up the cookie box and place it on the stove |
| 23 |  | spatial_dim3_change_object_ext_07_cookie_box_to_cabinet | pick up the cookie box and place it on the cabinet |
| 24 |  | spatial_dim3_change_object_ext_08_cookie_box_to_ramekin | pick up the cookie box and place it on the ramekin |
| 25 |  | spatial_dim5_drawer_action_ext_02_open_top | open the top drawer of the cabinet |
| 26 |  | spatial_dim5_drawer_action_ext_03_open_middle | open the middle drawer of the cabinet |
| 27 |  | spatial_dim5_drawer_action_ext_04_open_bottom | open the bottom drawer of the cabinet |
| 28 |  | goal_dim1_change_object_ext_02_wine_bottle_to_stove | put the wine bottle on the stove |
| 29 | O | goal_dim1_change_object_ext_03_cream_cheese_to_stove | put the cream cheese on the stove |
| 30 | O | goal_dim1_change_object_ext_05_cream_cheese_to_cabinet | put the cream cheese on top of the cabinet |
| 31 |  | goal_dim1_change_object_ext_06_wine_bottle_to_plate | put the wine bottle on the plate |
| 32 | O | goal_dim1_change_object_ext_07_cream_cheese_to_plate | put the cream cheese on the plate |
| 33 |  | goal_dim1_change_object_ext_09_cream_cheese_to_top_drawer | open the top drawer and put the cream cheese inside |
| 34 |  | goal_dim1_change_object_ext_11_bottom_drawer | open the bottom drawer of the cabinet |
| 35 |  | goal_dim1_change_object_ext_12_bowl_to_stove_front | put the bowl in front of the stove |
| 36 |  | goal_dim2_change_target_put_the_wine_bottle_in_front_of_the_stove | put the wine bottle in front of the stove |
| 37 |  | object_dim1_change_object_ext_01_scene0_salad_dressing | Pick the salad dressing and place it in the basket |
| 38 |  | object_dim1_change_object_ext_02_scene0_cream_cheese | Pick the cream cheese and place it in the basket |
| 39 | O | object_dim1_change_object_ext_03_scene1_alphabet_soup | Pick the alphabet soup and place it in the basket |
| 40 |  | object_dim1_change_object_ext_04_scene1_butter | Pick the butter and place it in the basket |
| 41 | O | object_dim1_change_object_ext_05_scene2_milk | Pick the milk and place it in the basket |
| 42 | O | object_dim1_change_object_ext_06_scene2_tomato_sauce | Pick the tomato sauce and place it in the basket |
| 43 |  | object_dim1_change_object_ext_07_scene3_ketchup | Pick the ketchup and place it in the basket |
| 44 |  | object_dim1_change_object_ext_08_scene3_chocolate_pudding | Pick the chocolate pudding and place it in the basket |
| 45 | O | object_dim1_change_object_ext_09_scene4_bbq_sauce | Pick the bbq sauce and place it in the basket |
| 46 |  | object_dim1_change_object_ext_10_scene4_milk | Pick the milk and place it in the basket |
| 47 |  | object_dim1_change_object_ext_11_scene5_bbq_sauce | Pick the bbq sauce and place it in the basket |
| 48 |  | object_dim1_change_object_ext_12_scene5_orange_juice | Pick the orange juice and place it in the basket |
| 49 |  | object_dim1_change_object_ext_13_scene6_ketchup | Pick the ketchup and place it in the basket |
| 50 |  | object_dim1_change_object_ext_14_scene6_tomato_sauce | Pick the tomato sauce and place it in the basket |
| 51 |  | object_dim1_change_object_ext_15_scene7_butter | Pick the butter and place it in the basket |
| 52 |  | object_dim1_change_object_ext_16_scene7_cream_cheese | Pick the cream cheese and place it in the basket |
| 53 |  | object_dim1_change_object_ext_17_scene8_orange_juice | Pick the orange juice and place it in the basket |
| 54 |  | object_dim1_change_object_ext_18_scene8_bbq_sauce | Pick the bbq sauce and place it in the basket |
| 55 |  | object_dim1_change_object_ext_19_scene8_ketchup | Pick the ketchup and place it in the basket |
| 56 |  | object_dim1_change_object_ext_20_scene9_chocolate_pudding | Pick the chocolate pudding and place it in the basket |
| 57 |  | object_dim1_change_object_ext_21_scene9_butter | Pick the butter and place it in the basket |
| 58 |  | object_dim1_change_object_ext_22_scene9_salad_dressing | Pick the salad dressing and place it in the basket |

trained 합계: 16/59