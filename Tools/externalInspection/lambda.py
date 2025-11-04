'''
Author       : kev1n
Date         : 2021-07-02 09:55:09
LastEditTime : 2021-07-02 15:24:00
a2V2MW4uemhlbmdAb3V0bG9vay5jb20=: 
'''

stair = [(1, 101629), (1, 102007)]
front = [(2, 101442), (2, 101701), (2, 101801)]
left = [(3, 101452), (3, 101621), (3, 101826), (3, 101922)]
tail = [(4, 101752), (4, 101930)]
right = [(5, 101840), (5, 101952)]


def isSubsequence(s: list, t: list) -> bool:
    n, m = len(s), len(t)
    i = j = 0
    while i < n and j < m:
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == n


sum = stair+front+left+tail+right
sortedsum = sorted(sum, key=lambda x: (x[1], x[0]))
sortedList = [item[0] for item in sortedsum]
subList = [1, 2, 3, 4, 5]

flag = isSubsequence(subList, sortedList)

print(flag)
