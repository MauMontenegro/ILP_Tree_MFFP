\* Firefighter_Tree *\
Maximize
Sum_of_Defended_Nodes: 15 Defend_1 + 3 Defend_10 + 3 Defend_11 + 3 Defend_12
 + 3 Defend_13 + 3 Defend_14 + Defend_15 + Defend_16 + Defend_17 + Defend_18
 + Defend_19 + 15 Defend_2 + Defend_20 + Defend_21 + Defend_22 + Defend_23
 + Defend_24 + Defend_25 + Defend_26 + Defend_27 + Defend_28 + Defend_29
 + 7 Defend_3 + Defend_30 + 7 Defend_4 + 7 Defend_5 + 7 Defend_6 + 3 Defend_7
 + 3 Defend_8 + 3 Defend_9
Subject To
Sum_of_Defended_level1: Defend_1 + Defend_2 <= 1
Sum_of_Defended_level2: Defend_3 + Defend_4 + Defend_5 + Defend_6 <= 1
Sum_of_Defended_level3: Defend_10 + Defend_11 + Defend_12 + Defend_13
 + Defend_14 + Defend_7 + Defend_8 + Defend_9 <= 1
Sum_of_Defended_level4: Defend_15 + Defend_16 + Defend_17 + Defend_18
 + Defend_19 + Defend_20 + Defend_21 + Defend_22 + Defend_23 + Defend_24
 + Defend_25 + Defend_26 + Defend_27 + Defend_28 + Defend_29 + Defend_30 <= 1
Sum_of_Leaf_Path15: Defend_1 + Defend_15 + Defend_3 + Defend_7 <= 1
Sum_of_Leaf_Path16: Defend_1 + Defend_16 + Defend_3 + Defend_7 <= 1
Sum_of_Leaf_Path17: Defend_1 + Defend_17 + Defend_3 + Defend_8 <= 1
Sum_of_Leaf_Path18: Defend_1 + Defend_18 + Defend_3 + Defend_8 <= 1
Sum_of_Leaf_Path19: Defend_1 + Defend_19 + Defend_4 + Defend_9 <= 1
Sum_of_Leaf_Path20: Defend_1 + Defend_20 + Defend_4 + Defend_9 <= 1
Sum_of_Leaf_Path21: Defend_1 + Defend_10 + Defend_21 + Defend_4 <= 1
Sum_of_Leaf_Path22: Defend_1 + Defend_10 + Defend_22 + Defend_4 <= 1
Sum_of_Leaf_Path23: Defend_11 + Defend_2 + Defend_23 + Defend_5 <= 1
Sum_of_Leaf_Path24: Defend_11 + Defend_2 + Defend_24 + Defend_5 <= 1
Sum_of_Leaf_Path25: Defend_12 + Defend_2 + Defend_25 + Defend_5 <= 1
Sum_of_Leaf_Path26: Defend_12 + Defend_2 + Defend_26 + Defend_5 <= 1
Sum_of_Leaf_Path27: Defend_13 + Defend_2 + Defend_27 + Defend_6 <= 1
Sum_of_Leaf_Path28: Defend_13 + Defend_2 + Defend_28 + Defend_6 <= 1
Sum_of_Leaf_Path29: Defend_14 + Defend_2 + Defend_29 + Defend_6 <= 1
Sum_of_Leaf_Path30: Defend_14 + Defend_2 + Defend_30 + Defend_6 <= 1
Binaries
Defend_1
Defend_10
Defend_11
Defend_12
Defend_13
Defend_14
Defend_15
Defend_16
Defend_17
Defend_18
Defend_19
Defend_2
Defend_20
Defend_21
Defend_22
Defend_23
Defend_24
Defend_25
Defend_26
Defend_27
Defend_28
Defend_29
Defend_3
Defend_30
Defend_4
Defend_5
Defend_6
Defend_7
Defend_8
Defend_9
End
