# LIBERO datasets — language commands (dataset × env/scene)

Source: `dataset_filtered/<ds>/meta/tasks.parquet` + LIBERO benchmark (scene parsed from bddl name).
`libero_spatial/object/goal` are single-scene suites (one env, varied along one axis).

## Summary

| dataset | suite | tasks | envs |
|---|---|---|---|
| `libero_90_full_full` | libero_90 | 73 | 20 |
| `libero_10_full_full` | libero_10 | 10 | 9 |
| `libero_spatial_full_full` | libero_spatial | 10 | 1 (varied spatial relations) |
| `libero_object_full_full` | libero_object | 10 | 1 (varied objects) |
| `libero_goal_full_full` | libero_goal | 10 | 1 (varied goals) |

## libero_90_full_full  ·  libero_90  ·  73 tasks  ·  20 env(s)

### KITCHEN_SCENE1  (2)
- `[62]` open the top drawer of the cabinet and put the bowl in it
- `[72]` open the bottom drawer of the cabinet

### KITCHEN_SCENE10  (4)
- `[18]` put the butter at the back in the top drawer of the cabinet and close it
- `[23]` put the chocolate pudding in the top drawer of the cabinet and close it
- `[26]` close the top drawer of the cabinet and put the black bowl on top of it
- `[32]` put the butter at the front in the top drawer of the cabinet and close it

### KITCHEN_SCENE2  (7)
- `[5]` put the middle black bowl on the plate
- `[14]` put the middle black bowl on top of the cabinet
- `[34]` put the black bowl at the front on the plate
- `[38]` stack the middle black bowl on the back black bowl
- `[55]` put the black bowl at the back on the plate
- `[61]` open the top drawer of the cabinet
- `[67]` stack the black bowl at the front on the black bowl in the middle

### KITCHEN_SCENE3  (2)
- `[64]` put the frying pan on the stove
- `[69]` put the moka pot on the stove

### KITCHEN_SCENE4  (5)
- `[7]` put the wine bottle on the wine rack
- `[28]` put the wine bottle in the bottom drawer of the cabinet
- `[29]` close the bottom drawer of the cabinet and open the top drawer
- `[39]` close the bottom drawer of the cabinet
- `[44]` put the black bowl in the bottom drawer of the cabinet

### KITCHEN_SCENE5  (5)
- `[4]` put the black bowl on top of the cabinet
- `[8]` put the black bowl in the top drawer of the cabinet
- `[31]` put the ketchup in the top drawer of the cabinet
- `[66]` put the black bowl on the plate
- `[70]` close the top drawer of the cabinet

### KITCHEN_SCENE6  (2)
- `[17]` close the microwave
- `[41]` put the yellow and white mug to the front of the white mug

### KITCHEN_SCENE7  (3)
- `[10]` put the white bowl on the plate
- `[20]` open the microwave
- `[63]` put the white bowl to the right of the plate

### KITCHEN_SCENE8  (2)
- `[13]` put the right moka pot on the stove
- `[71]` turn off the stove

### KITCHEN_SCENE9  (6)
- `[6]` put the white bowl on top of the cabinet
- `[11]` turn on the stove and put the frying pan on it
- `[16]` put the frying pan under the cabinet shelf
- `[30]` put the frying pan on the cabinet shelf
- `[43]` turn on the stove
- `[56]` put the frying pan on top of the cabinet

### LIVING_ROOM_SCENE1  (2)
- `[45]` pick up the cream cheese box and put it in the basket
- `[60]` pick up the ketchup and put it in the basket

### LIVING_ROOM_SCENE2  (4)
- `[1]` pick up the alphabet soup and put it in the basket
- `[3]` pick up the tomato sauce and put it in the basket
- `[27]` pick up the orange juice and put it in the basket
- `[49]` pick up the milk and put it in the basket

### LIVING_ROOM_SCENE3  (5)
- `[2]` pick up the cream cheese and put it in the tray
- `[46]` pick up the tomato sauce and put it in the tray
- `[48]` pick up the alphabet soup and put it in the tray
- `[53]` pick up the ketchup and put it in the tray
- `[59]` pick up the butter and put it in the tray

### LIVING_ROOM_SCENE4  (5)
- `[33]` pick up the chocolate pudding and put it in the tray
- `[36]` stack the right bowl on the left bowl and place them in the tray
- `[50]` stack the left bowl on the right bowl and place them in the tray
- `[57]` pick up the salad dressing and put it in the tray
- `[65]` pick up the black bowl on the left and put it in the tray

### LIVING_ROOM_SCENE5  (4)
- `[9]` put the yellow and white mug on the right plate
- `[19]` put the red mug on the left plate
- `[52]` put the white mug on the left plate
- `[58]` put the red mug on the right plate

### LIVING_ROOM_SCENE6  (4)
- `[15]` put the red mug on the plate
- `[24]` put the chocolate pudding to the right of the plate
- `[25]` put the chocolate pudding to the left of the plate
- `[42]` put the white mug on the plate

### STUDY_SCENE1  (1)
- `[21]` pick up the yellow and white mug and place it to the right of the caddy

### STUDY_SCENE2  (1)
- `[37]` pick up the book and place it in the back compartment of the caddy

### STUDY_SCENE3  (5)
- `[0]` pick up the book and place it in the right compartment of the caddy
- `[12]` pick up the book and place it in the front compartment of the caddy
- `[35]` pick up the book and place it in the left compartment of the caddy
- `[47]` pick up the white mug and place it to the right of the caddy
- `[51]` pick up the red mug and place it to the right of the caddy

### STUDY_SCENE4  (4)
- `[22]` pick up the book in the middle and place it on the cabinet shelf
- `[40]` pick up the book on the right and place it under the cabinet shelf
- `[54]` pick up the book on the left and place it on top of the shelf
- `[68]` pick up the book on the right and place it on the cabinet shelf

## libero_10_full_full  ·  libero_10  ·  10 tasks  ·  9 env(s)

### KITCHEN_SCENE3  (1)
- `[3]` turn on the stove and put the moka pot on it

### KITCHEN_SCENE4  (1)
- `[8]` put the black bowl in the bottom drawer of the cabinet and close it

### KITCHEN_SCENE6  (1)
- `[2]` put the yellow and white mug in the microwave and close it

### KITCHEN_SCENE8  (1)
- `[6]` put both moka pots on the stove

### LIVING_ROOM_SCENE1  (1)
- `[4]` put both the alphabet soup and the cream cheese box in the basket

### LIVING_ROOM_SCENE2  (2)
- `[5]` put both the alphabet soup and the tomato sauce in the basket
- `[7]` put both the cream cheese box and the butter in the basket

### LIVING_ROOM_SCENE5  (1)
- `[0]` put the white mug on the left plate and put the yellow and white mug on the right plate

### LIVING_ROOM_SCENE6  (1)
- `[1]` put the white mug on the plate and put the chocolate pudding to the right of the plate

### STUDY_SCENE1  (1)
- `[9]` pick up the book and place it in the back compartment of the caddy

## libero_spatial_full_full  ·  libero_spatial  ·  10 tasks  ·  1 env(s)

### libero_spatial · single scene  (10)
- `[0]` pick up the black bowl next to the cookie box and place it on the plate
- `[1]` pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
- `[2]` pick up the black bowl on the ramekin and place it on the plate
- `[3]` pick up the black bowl on the stove and place it on the plate
- `[4]` pick up the black bowl between the plate and the ramekin and place it on the plate
- `[5]` pick up the black bowl on the cookie box and place it on the plate
- `[6]` pick up the black bowl next to the plate and place it on the plate
- `[7]` pick up the black bowl next to the ramekin and place it on the plate
- `[8]` pick up the black bowl from table center and place it on the plate
- `[9]` pick up the black bowl on the wooden cabinet and place it on the plate

## libero_object_full_full  ·  libero_object  ·  10 tasks  ·  1 env(s)

### libero_object · single scene  (10)
- `[0]` pick up the orange juice and place it in the basket
- `[1]` pick up the ketchup and place it in the basket
- `[2]` pick up the cream cheese and place it in the basket
- `[3]` pick up the bbq sauce and place it in the basket
- `[4]` pick up the alphabet soup and place it in the basket
- `[5]` pick up the milk and place it in the basket
- `[6]` pick up the salad dressing and place it in the basket
- `[7]` pick up the butter and place it in the basket
- `[8]` pick up the tomato sauce and place it in the basket
- `[9]` pick up the chocolate pudding and place it in the basket

## libero_goal_full_full  ·  libero_goal  ·  10 tasks  ·  1 env(s)

### libero_goal · single scene  (10)
- `[0]` put the bowl on the plate
- `[1]` put the wine bottle on the rack
- `[2]` open the top drawer and put the bowl inside
- `[3]` put the cream cheese in the bowl
- `[4]` put the wine bottle on top of the cabinet
- `[5]` push the plate to the front of the stove
- `[6]` turn on the stove
- `[7]` put the bowl on the stove
- `[8]` put the bowl on top of the cabinet
- `[9]` open the middle drawer of the cabinet
