def coinChange2(coins , amount: int) -> int:
    answer = []

    def helper(index, lst, total):
        if total == amount:
            answer.append(list(lst))
            return
        if total > amount:
            return
        for i in range(index, len(coins)):
            lst.append(coins[i])
            helper(i, lst, total + coins[i])
            lst.pop()

    helper(0, [], 0)
    return answer


if __name__ == '__main__':
    print(coinChange2([1,2], 5))