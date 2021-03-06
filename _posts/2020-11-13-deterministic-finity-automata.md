---
layout: post
title: 确定的有限状态机 （DFA）——解决有效数字问题
categories: [算法和数据结构]
description: leetcode
keywords: 有限状态机，leetcode
---

### 确定的有限状态机
一个确定的有限状态机 $M$ 是一个 5 元组，$(Q, \Sigma, \delta, q_0, F)$，包括：
- 状态的有限集合 $Q$
- 符号的有限集合，字母表 $\Sigma$
- 转移函数，转移表 $\delta$：$Q \times \Sigma \rightarrow Q$
- 一个开始状态 $q_0 \in Q$
- 可接受状态的集合 $F \in Q$

让 $w=a_1a_2...a_n$ 表示在字母表 $\Sigma$ 上的一个字符串，当状态序列 $r_0，r_1,...,r_n$ 满足如下条件时，有限状态机接受字符串 $w$:
1. $r_0=q_0$
2. $r_{i+1} = \delta(r_i, a_{i+1})$
3. $r_n \in F$

第一个条件表示状态机的初试状态为 $q_0$。第二个条件根据当前的状态和接受的字符 a_{i+1}，通过转移表更新当前的状态。最后一个条件表示如果最后的状态属于接受状态 $F$，那么自动机接受字符串 $w$，否则拒绝它。状态机 $M$ 接受的字符串组成一个语言 $L(M)$。