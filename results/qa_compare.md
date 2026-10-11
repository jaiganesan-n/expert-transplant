# Questions and answers: stage1_test → stage2_5m → stage3_20m

Each question answered in one sentence (few-shot frame, greedy). Gemma is the original model, No FFN is Gemma with all 26 FFNs deleted, and each run is the hybrid trained in that run.

| Model | Tokens trained | Correct | Fact log-prob | Fact recovery | Sentence log-prob | Copied | Mix perplexity | GR_CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No FFN (floor) | – | 0/30 | -14.81 | 0% | -12.91 | 0/30 | – | – |
| Hybrid, stage1_test | 0.98M | 1/30 | -5.33 | 65% | -2.29 | 11/30 | 71.3 | 0.852 |
| Hybrid, stage2_5m | 5.00M | 1/30 | -3.18 | 80% | -1.22 | 0/30 | 34.0 | 0.909 |
| Hybrid, stage3_20m | 19.99M | 6/30 | -1.90 | 89% | -0.92 | 0/30 | 23.5 | 0.937 |
| Gemma (target) | – | 28/30 | -0.23 | 100% | -0.28 | 0/30 | – | – |

From stage1_test to stage3_20m, the fact got more likely for 29 of 30 questions and less likely for 1.

## Fact log-prob per question

| # | Question | Topic | Gemma | No FFN | stage1_test | stage2_5m | stage3_20m | Change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | What is the capital of France? | geography | -0.0 ✓ | -14.0 ✗ | -5.0 ✗ | -2.4 ✗ | -1.0 ✓ | +4.0 |
| 2 | What is the capital of Japan? | geography | -0.0 ✓ | -17.6 ✗ | -8.2 ✗ | -4.4 ✗ | -1.7 ✓ | +6.5 |
| 3 | What is the capital of Germany? | geography | -0.0 ✓ | -16.9 ✗ | -10.3 ✗ | -5.5 ✗ | -2.9 ✗ | +7.4 |
| 4 | In which country are the pyramids of Giza? | geography | -0.0 ✓ | -18.6 ✗ | -6.3 ✗ | -3.3 ✗ | -1.8 ✗ | +4.6 |
| 5 | What language is spoken in Brazil? | geography | -0.0 ✓ | -18.6 ✗ | -5.6 ✗ | -1.7 ✗ | -0.6 ✓ | +4.9 |
| 6 | What is the longest river in Africa? | geography | -0.1 ✓ | -14.5 ✗ | -8.1 ✗ | -3.0 ✗ | -2.8 ✗ | +5.3 |
| 7 | How many continents are there? | geography | -0.4 ✓ | -10.1 ✗ | -3.3 ✗ | -2.5 ✗ | -1.8 ✗ | +1.4 |
| 8 | What gas do plants absorb from the air? | science | -0.3 ✓ | -12.4 ✗ | -4.5 ✗ | -3.3 ✗ | -1.7 ✗ | +2.8 |
| 9 | What is the chemical formula of water? | science | -0.1 ✓ | -12.9 ✗ | -6.7 ✗ | -4.6 ✗ | -2.0 ✗ | +4.7 |
| 10 | At what temperature in Celsius does water freeze? | science | -0.3 ✓ | -9.9 ✗ | -2.9 ✗ | -2.0 ✗ | -1.5 ✗ | +1.4 |
| 11 | Which planet is known as the Red Planet? | science | -0.2 ✓ | -17.6 ✗ | -8.5 ✗ | -7.7 ✗ | -4.1 ✗ | +4.4 |
| 12 | What is the largest planet in our solar system? | science | -0.1 ✓ | -20.7 ✗ | -8.8 ✗ | -2.3 ✗ | -1.6 ✓ | +7.2 |
| 13 | What star is at the center of our solar system? | science | -0.4 ✓ | -15.6 ✗ | -4.3 ✓ | -2.2 ✗ | -0.8 ✗ | +3.5 |
| 14 | What organ pumps blood through the human body? | science | -0.0 ✓ | -17.5 ✗ | -6.1 ✗ | -3.8 ✗ | -2.0 ✗ | +4.1 |
| 15 | What force pulls objects toward the Earth? | science | -0.6 ✓ | -21.8 ✗ | -6.3 ✗ | -3.6 ✗ | -2.0 ✓ | +4.3 |
| 16 | What is 2 + 2? | math | -0.0 ✓ | -8.8 ✗ | -2.0 ✗ | -1.6 ✗ | -2.2 ✗ | -0.2 |
| 17 | What is the square root of 81? | math | -1.6 ✗ | -9.2 ✗ | -3.0 ✗ | -2.2 ✗ | -1.7 ✗ | +1.3 |
| 18 | How many sides does a triangle have? | math | -0.0 ✓ | -9.4 ✗ | -3.0 ✗ | -2.8 ✗ | -1.9 ✗ | +1.1 |
| 19 | What is 10 times 10? | math | -0.0 ✓ | -11.0 ✗ | -1.6 ✗ | -0.8 ✗ | -0.7 ✗ | +0.9 |
| 20 | How many days are in a week? | everyday | -0.4 ✓ | -10.1 ✗ | -2.9 ✗ | -2.1 ✗ | -1.8 ✗ | +1.1 |
| 21 | How many months are in a year? | everyday | -0.1 ✓ | -9.5 ✗ | -2.6 ✗ | -1.5 ✗ | -0.9 ✗ | +1.7 |
| 22 | Which day comes after Monday? | everyday | -0.4 ✓ | -18.4 ✗ | -7.6 ✗ | -5.3 ✗ | -3.4 ✗ | +4.2 |
| 23 | What color is the sky on a clear day? | everyday | -0.4 ✓ | -16.7 ✗ | -3.7 ✗ | -2.0 ✓ | -1.3 ✓ | +2.4 |
| 24 | What is the opposite of hot? | everyday | -0.1 ✓ | -14.7 ✗ | -7.6 ✗ | -3.9 ✗ | -3.0 ✗ | +4.6 |
| 25 | Which animal is known as the king of the jungle? | everyday | -0.3 ✓ | -16.9 ✗ | -3.7 ✗ | -4.1 ✗ | -1.3 ✗ | +2.4 |
| 26 | What is the main ingredient of guacamole? | everyday | -1.0 ✗ | -19.3 ✗ | -6.4 ✗ | -4.0 ✗ | -2.8 ✗ | +3.6 |
| 27 | Who wrote Romeo and Juliet? | history | -0.1 ✓ | -17.3 ✗ | -3.8 ✗ | -2.8 ✗ | -1.7 ✗ | +2.1 |
| 28 | Who was the first president of the United States? | history | -0.0 ✓ | -14.9 ✗ | -6.5 ✗ | -3.2 ✗ | -2.3 ✗ | +4.2 |
| 29 | Who was the first person to walk on the Moon? | history | -0.0 ✓ | -17.6 ✗ | -8.8 ✗ | -5.4 ✗ | -2.8 ✗ | +6.0 |
| 30 | In which year did World War II end? | history | -0.0 ✓ | -11.9 ✗ | -2.1 ✗ | -1.6 ✗ | -1.1 ✗ | +1.0 |

✓ / ✗: whether that model's reply contained the answer. Change: last run minus first run.

## Replies

**1. What is the capital of France?** (expected: The capital of France is Paris.)

- Gemma: The capital of France is Paris. ✓
- No FFN: (nothing) ✗
- stage1_test: The capital of France is a capital of France. ✗
- stage2_5m: The capital of France is France. ✗
- stage3_20m: The capital of France is Paris. ✓

**2. What is the capital of Japan?** (expected: The capital of Japan is Tokyo.)

- Gemma: The capital of Japan is Tokyo. ✓
- No FFN: (nothing) ✗
- stage1_test: The Mona Lisa was painted by a black black black black black black black black black black black black black black black black black black black black black black black black black black black black black black black black black ✗
- stage2_5m: The capital of Japan is a capital of the United States. ✗
- stage3_20m: The capital of Japan is Tokyo. ✓

**3. What is the capital of Germany?** (expected: The capital of Germany is Berlin.)

- Gemma: The capital of Germany is Berlin. ✓
- No FFN: (nothing) ✗
- stage1_test: The largest ocean on Earth is the largest ocean. ✗
- stage2_5m: The capital of Germany is the capital of Germany. ✗
- stage3_20m: The capital of Germany is the capital of the United States. ✗

**4. In which country are the pyramids of Giza?** (expected: The pyramids of Giza are in Egypt.)

- Gemma: The pyramids of Giza are in Egypt. ✓
- No FFN: (nothing) ✗
- stage1_test: Bees make honey. ✗
- stage2_5m: The pyramids of Giza are the largest country in the United States. ✗
- stage3_20m: Giza is in the United States. ✗

**5. What language is spoken in Brazil?** (expected: The main language spoken in Brazil is Portuguese.)

- Gemma: The language spoken in Brazil is Portuguese. ✓
- No FFN: (nothing) ✗
- stage1_test: What language is spoken in Brazil? ✗
- stage2_5m: The language spoken in Brazil is spoken in Brazil. ✗
- stage3_20m: The language spoken in Brazil is Portuguese. ✓

**6. What is the longest river in Africa?** (expected: The longest river in Africa is the Nile.)

- Gemma: The longest river in Africa is the Nile. ✓
- No FFN: (nothing) ✗
- stage1_test: How many people are in Africa? ✗
- stage2_5m: The longest river in Africa is the Congo. ✗
- stage3_20m: The longest river in Africa is the Congo River. ✗

**7. How many continents are there?** (expected: There are seven continents.)

- Gemma: There are seven continents. ✓
- No FFN: (nothing) ✗
- stage1_test: How many people are there? ✗
- stage2_5m: The continents are continents. ✗
- stage3_20m: There are 12 continents. ✗

**8. What gas do plants absorb from the air?** (expected: Plants absorb carbon dioxide from the air.)

- Gemma: Plants absorb carbon dioxide from the air. ✓
- No FFN: (nothing) ✗
- stage1_test: What is the sun? ✗
- stage2_5m: Plants absorb gas from the air. ✗
- stage3_20m: The gas that plants absorb from the air is called gas. ✗

**9. What is the chemical formula of water?** (expected: The chemical formula of water is H2O.)

- Gemma: Water is H2O. ✓
- No FFN: (nothing) ✗
- stage1_test: What is the chemical formula of water? ✗
- stage2_5m: The chemical formula of water is the formula of water. ✗
- stage3_20m: Water is a chemical formula. ✗

**10. At what temperature in Celsius does water freeze?** (expected: Water freezes at 0 degrees Celsius.)

- Gemma: Water freezes at 0°C. ✓
- No FFN: (nothing) ✗
- stage1_test: The temperature in the sun is the temperature in the sun. ✗
- stage2_5m: Water freezes at 100 degrees Celsius. ✗
- stage3_20m: Water freezes at 10 degrees Celsius. ✗

**11. Which planet is known as the Red Planet?** (expected: The Red Planet is Mars.)

- Gemma: The planet Mars is known as the Red Planet. ✓
- No FFN: (nothing) ✗
- stage1_test: The Red Planet is the largest ocean on Earth. ✗
- stage2_5m: The Red Planet is known as the Red Planet. ✗
- stage3_20m: The Red Planet is the planet that was discovered by the Italian explorer Giovanni Giovanni Giovanni. ✗

**12. What is the largest planet in our solar system?** (expected: The largest planet in our solar system is Jupiter.)

- Gemma: The largest planet in our solar system is Jupiter. ✓
- No FFN: (nothing) ✗
- stage1_test: How many people are in the planet? ✗
- stage2_5m: A planet is the largest planet in our solar system. ✗
- stage3_20m: The largest planet in our solar system is Jupiter. ✓

**13. What star is at the center of our solar system?** (expected: The star at the center of our solar system is the Sun.)

- Gemma: The star at the center of our solar system is the sun. ✓
- No FFN: (nothing) ✗
- stage1_test: What is the center of the sun is at the center of the center? ✓
- stage2_5m: A star is at the center of our solar system. ✗
- stage3_20m: The star is at the center of our solar system. ✗

**14. What organ pumps blood through the human body?** (expected: Blood is pumped through the human body by the heart.)

- Gemma: The heart pumps blood through the human body. ✓
- No FFN: (nothing) ✗
- stage1_test: How many people are living in the world? ✗
- stage2_5m: The organ pumps blood. ✗
- stage3_20m: The organ pumps blood through the body. ✗

**15. What force pulls objects toward the Earth?** (expected: Objects are pulled toward the Earth by gravity.)

- Gemma: The force that pulls objects toward the Earth is gravity. ✓
- No FFN: (nothing) ✗
- stage1_test: What is the force pulling the objects to the Earth? ✗
- stage2_5m: The force pulls objects toward the Earth. ✗
- stage3_20m: The Earth pulls the force of gravity. ✓

**16. What is 2 + 2?** (expected: 2 + 2 equals 4.)

- Gemma: 4. ✓
- No FFN: (nothing) ✗
- stage1_test: Bees make honey. ✗
- stage2_5m: 2 + 2. ✗
- stage3_20m: 2 + 2 is 2. ✗

**17. What is the square root of 81?** (expected: The square root of 81 is 9.)

- Gemma: The square root of 81 is 13. ✗
- No FFN: (nothing) ✗
- stage1_test: What is the square root of 81? ✗
- stage2_5m: The square root of 81 is the square root of 81. ✗
- stage3_20m: The square root of 81 is 81. ✗

**18. How many sides does a triangle have?** (expected: A triangle has three sides.)

- Gemma: A triangle has three sides. ✓
- No FFN: (nothing) ✗
- stage1_test: A spider has eight legs. ✗
- stage2_5m: A triangle has two sides. ✗
- stage3_20m: A triangle has 12 sides. ✗

**19. What is 10 times 10?** (expected: 10 times 10 equals 100.)

- Gemma: 100. ✓
- No FFN: (nothing) ✗
- stage1_test: What is 10 times 10? ✗
- stage2_5m: 10 times 10. ✗
- stage3_20m: 10 times 10. ✗

**20. How many days are in a week?** (expected: There are seven days in a week.)

- Gemma: There are seven days in a week. ✓
- No FFN: (nothing) ✗
- stage1_test: How many days are in a week? ✗
- stage2_5m: The Mona Lisa was in a week. ✗
- stage3_20m: The Mona Lisa was in a week. ✗

**21. How many months are in a year?** (expected: There are twelve months in a year.)

- Gemma: There are 12 months in a year. ✓
- No FFN: (nothing) ✗
- stage1_test: How many people are in a year? ✗
- stage2_5m: The largest month is the Pacific Ocean. ✗
- stage3_20m: The number of months in a year is 1000. ✗

**22. Which day comes after Monday?** (expected: The day after Monday is Tuesday.)

- Gemma: The day after Monday is Tuesday. ✓
- No FFN: (nothing) ✗
- stage1_test: The most people are in the world, and the most people are in the world. ✗
- stage2_5m: The day comes after Monday. ✗
- stage3_20m: The day after Monday comes after Monday. ✗

**23. What color is the sky on a clear day?** (expected: On a clear day the sky is blue.)

- Gemma: The sky is blue on a clear day. ✓
- No FFN: (nothing) ✗
- stage1_test: The sky on a day is the sky. ✗
- stage2_5m: The sky is a blue sky. ✓
- stage3_20m: The sky is blue. ✓

**24. What is the opposite of hot?** (expected: The opposite of hot is cold.)

- Gemma: Cold. ✓
- No FFN: (nothing) ✗
- stage1_test: A spider has eight legs. ✗
- stage2_5m: A hot is hot. ✗
- stage3_20m: The opposite of hot is the opposite of hot. ✗

**25. Which animal is known as the king of the jungle?** (expected: The animal known as the king of the jungle is the lion.)

- Gemma: The king of the jungle is a lion. ✓
- No FFN: (nothing) ✗
- stage1_test: Grass is green. ✗
- stage2_5m: The king of the jungle is the king of the jungle. ✗
- stage3_20m: The king of the jungle is the king of the jungle. ✗

**26. What is the main ingredient of guacamole?** (expected: The main ingredient of guacamole is avocado.)

- Gemma: Guacamole is made of avocados. ✗
- No FFN: (nothing) ✗
- stage1_test: The Mona Lisa was painted by a red. ✗
- stage2_5m: The main ingredient of guacamole is guacamole. ✗
- stage3_20m: The main ingredient of guacamole is guacamole. ✗

**27. Who wrote Romeo and Juliet?** (expected: Romeo and Juliet was written by William Shakespeare.)

- Gemma: Romeo and Juliet was written by William Shakespeare. ✓
- No FFN: (nothing) ✗
- stage1_test: What is the largest ocean on Earth? ✗
- stage2_5m: Romeo and Juliet were written by Romeo. ✗
- stage3_20m: The Romeo and Juliet were written by Romeo and Juliet. ✗

**28. Who was the first president of the United States?** (expected: The first president of the United States was George Washington.)

- Gemma: The first president of the United States was George Washington. ✓
- No FFN: (nothing) ✗
- stage1_test: The president of the United States was the first president of the United States. ✗
- stage2_5m: The president of the United States is the president of the United States. ✗
- stage3_20m: The president of the United States was George W. Bush. ✗

**29. Who was the first person to walk on the Moon?** (expected: The first person to walk on the Moon was Neil Armstrong.)

- Gemma: The first person to walk on the Moon was Neil Armstrong. ✓
- No FFN: (nothing) ✗
- stage1_test: What is the first person to walk on the moon? ✗
- stage2_5m: The first person to walk on the Moon is the first person. ✗
- stage3_20m: The first person to walk on the Moon is the Apollo. ✗

**30. In which year did World War II end?** (expected: World War II ended in 1945.)

- Gemma: World War II ended in 1945. ✓
- No FFN: (nothing) ✗
- stage1_test: The World War II was a part of the world. ✗
- stage2_5m: The World War II ended in 1999. ✗
- stage3_20m: The War II ended in 1939. ✗

Fact log-prob: mean log-probability per token of the fact after the reference sentence's opening ("The capital of France is" -> "Paris"); 0 is certain, higher is better. Fact recovery: the hybrid's fact log-prob between No FFN (0%) and Gemma (100%), the question-answering counterpart of GR_CE.
