# Identity

You are a program repair system that helps to improve program in Python3 code.


# Instructions
You will be given a 'Target Program' along with a 'Description' of its intent, a set of 'Test Results', static analysis 'Warning Messages', and optionally a 'Reference Program'.
Use all of this information to fix the program, following the fix guidelines:
* Fix guidelines:
  - Pass all test cases
  - Minimally changes the program syntax
  - Minimally changes the program behavior
  - Reduce memory usage
  - Reduce runtime
  - Improve readability
  - Improve quality
  - Improve maintainability
* Print the given structure.


# Example

## Input

### Target Program

```python
def search(x, seq):
    for i in range(len(seq)):
        if seq[i] == x:
            return i
    return -1

```

### Description

This function searches for the first occurrence of `x` in the sequence `seq` and returns its index.

### Test Results

|  #  | Input | Expected | Actual | Status |
| :-: | :---: | :------: | :----: | :----: |
|  1  | search(42, (-5, 1, 3, 5, 7, 10)) | 6 | -1 | Failure |
|  2  | search(5, [1, 2, 3, 4]) | 3 | -1 | Failure |
|  3  | search(100, []) | 0 | -1 | Failure |

### Warning Messages

* Line: 1 -> C0114:Missing module docstring
* Line: 1 -> C0116:Missing function or method docstring
* Line: 2 -> C0200:Consider using enumerate instead of iterating with range and len
* Line: 6 -> C0305:Trailing newlines
* Maintainability Score: 76.89330965566353
* Function: search -> Maintainability Rank: A

### Reference Program

```python
def search(x, seq):
    for i, ele in enumerate(seq):
        if x <= ele:
            return i
    return len(seq)
```
## Output

### Fixed Program

```python
"""Utility module for sequence operations."""

def search(x, seq):
    """Return the index of x in seq, or len(seq)"""
    for i, ele in enumerate(seq):
        if ele >= x:
            return i
    return len(seq)
```

## Inputs

### Target Program

```python
n, k = map(int, input().split())
l = list(map(int, input().split()))
p = []
c = 0
for i in range(n - 1):
    if l[i] > 0 and l[i + 1] < 0 or (l[i] < 0 and l[i + 1] > 0):
        c += 1
        p.append(i + 1)
        l[i] = -l[i]
        l[i + 1] = -l[i + 1]
if c > 0:
    print(len(p))
    print(' '.join(map(str, p)))
else:
    print(0)
```

### Description
Let us play a game of cards. As you all know there are 52 cards in a regular deck, but our deck is somewhat different. Our deck consists of N cards. Each card simply contains a unique number from 1 to N. Initially all the cards are kept on the table, facing up. Rob, the notorious kid, flipped some of the cards, i.e. made them face down. Now, your task is to print any valid sequence of magic operations (as defined below) such that after a maximum of K magic operations, all the cards are facing upwards only.
A magic operation on card at position i consists of flipping its direction from up to down and vice versa. Also, if there exists a card on its right side, i.e. at position (i+1), that is also flipped.
Print the index of the cards that you are going to perform the magic operation on.
Input
Input consists of 2 lines. The first line contains 2 integers N and K, denoting the number of cards in our deck and the maximum possible number of moves allowed.
The second line contains N integers, whose absolute value lies from 1 to N. If a number is less than 0, it is initially facing down, else it is facing upwards.
It is guaranteed that all the absolute values on the cards are unique.
Output
Output on the first line an integer X, denoting the number of operations you intend to make for making all the cards face upwards. This integer should be less than or equal to K.
On the next line of the output, print X space separated integers, denoting the order in which you want to carry on the magic operations on any of the N cards.
Note: You have to print the index of the cards that you are going to perform the magic operation on.
If there exist multiple solutions, output any of them
Constraints
1 ≤ N ≤ 1000
N ≤ K ≤ max(5000, N * N)
Sample 1:
Input
Output
3 4
1 2 3
0
Explanation:
All the cards are already facing up.
Sample 2:
Input
Output
5 12
4 -2 3 -1 5
2
2 3
Explanation:
Below sequence shows the cards after every magic operation:
t = 0,    4 -2  3 -1 5
t = 1,    4  2 -3 -1 5
t = 2,    4  2  3  1 5
Finally all the cards are now facing upwards and the number of steps are also less than k i.e. 12.


### Test Results
|  #  | Input | Expected | Actual | Status |
| :-: | :---: | :------: | :----: | :----: |
|  1  | 3 4<br />1 2 3 | 0 | 0 | Success |
|  2  | 5 12<br />4 -2 3 -1 5 | 2<br />2 3 | 2<br />1 3 | Failure |
|  3  | 10 78<br />4 2 3 -9 -5 10 1 6 -7 8 | 3<br />4 9 10 | 3<br />3 4 8 | Failure |
|  4  | 4 5<br />1 -2 -3 4 | 1<br />2 | 2<br />1 2 | Failure |
|  5  | 7 15<br />-1 2 -3 4 -5 6 -7 | 4<br />1 2 5 6 | 3<br />1 3 5 | Failure |

### Warning Messages
* Line: 15 -> C0304:Final newline missing
* Line: 1 -> C0114:Missing module docstring
* Line: 4 -> C0103:Constant name "c" doesn't conform to UPPER_CASE naming style
* Line: 6 -> R1716:Simplify chained comparison between the operands
* Line: 6 -> R1716:Simplify chained comparison between the operands
* Maintainability Score: 56.93781004188902
* Maintainability Rank: A

### Reference Program

```python
n, k = map(int, input().split())
l = list(map(int, input().split()))
p = []
c = 0
for i in range(n - 1):
    if l[i] > 0 and l[i + 1] < 0 or (l[i] < 0 and l[i + 1] > 0):
        c += 1
        p.append(i + 1)
        l[i] = -l[i]
        l[i + 1] = -l[i + 1]
if c > 0:
    print(len(p))
    print(' '.join(map(str, p)))
else:
    print(0)
```