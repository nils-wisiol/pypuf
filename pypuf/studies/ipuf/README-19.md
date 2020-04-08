# Recovery of Experiment Information for (1,9) 64-bit iPUFs

Due to a two week hard limit on run time on the cluster, none of our (1,9) experiments made it to a regular finish. 
However, from log files we recovered the following information.

Out of 100 scheduled experiments, 77 failed due to technical errors.

At the time of cancellation, the status of the remaining 23 experiments was as follows:

seed |    start          |   duration   | last known accuracy | accuracy up (1) |   accuracy down (1)
---- | ----------------- |  ----------- | -------- | --------------- | ----------------------
0    | 03/12/20 10:41    |    304:59:00|   0.9954  |         
1    | 03/12/20 10:41    |    277:56:39|   0.9936  |          
8    | 03/12/20 10:41    |    335:17:46|   0.8756 (2)  | 0.00 |  0.87
42   | 03/12/20 10:49    |    335:16:27|   0.8692 (2)  | 0.00 |  0.01
7    | 03/12/20 10:41    |    335:32:15|   0.8592 (2)  | 1.00 |  0.99
41   | 03/12/20 10:49    |    335:28:33|   0.85 (2)    | 0.00 |  0.9
2    | 03/12/20 10:41    |    335:56:48|   ~0.5  | ~0.5 | ~0.5         
3    | 03/12/20 10:41    |    335:29:56|   ~0.5  | ~0.5 | ~0.5         
4    | 03/12/20 10:41    |    335:42:43|   ~0.5  | ~0.5 | ~0.5         
6    | 03/12/20 10:41    |    335:37:20|   ~0.5  | ~0.5 | ~0.5         
9    | 03/12/20 10:41    |    335:35:28|   ~0.5  | ~0.5 | ~0.5         
10   | 03/12/20 10:41    |    335:32:56|   ~0.5  | ~0.5 | ~0.5         
17   | 03/12/20 10:42    |    335:42:38|   ~0.5  | ~0.5 | ~0.5         
18   | 03/12/20 10:42    |    335:45:47|   ~0.5  | ~0.5 | ~0.5         
23   | 03/12/20 10:42    |    334:44:52|   ~0.5  | ~0.5 | ~0.5         
28   | 03/12/20 10:44    |    335:58:24|   ~0.5  | ~0.5 | ~0.5         
29   | 03/12/20 10:44    |    335:55:54|   ~0.5  | ~0.5 | ~0.5         
46   | 03/12/20 10:50    |    335:05:23|   ~0.5  | ~0.5 | ~0.5         
47   | 03/12/20 10:50    |    335:45:14|   ~0.5  | ~0.5 | ~0.5         
58   | 03/12/20 10:55    |    335:44:41|   ~0.5  | ~0.5 | ~0.5         
65   | 03/12/20 10:58    |    335:55:37|   ~0.5  | ~0.5 | ~0.5         
66   | 03/12/20 10:58    |    335:29:02|   ~0.5  | ~0.5 | ~0.5         
22   | 03/12/20 10:42    |    335:29:56|   ~0.5  | ~0.5 | ~0.5         
 
(1) Note that this is training set accuracy, but approximation of actual accuracy can be assumed.
(2) Based on newer known training set accuracies, this value can be assumed outdated.

Based on the status of the attack at the time of cancellation, we can safely assume that the top 6 experiments could
have made it to a successful final result, given more time or better programming.
