def convert(allocation=[32,32,32,32],capacity=32):
        allocation_mask=[]
        for i in allocation:
            my_list_true = [True for _ in range(i)]
            my_list_False=[False for _ in range(capacity-i)]
            allocation_mask.extend(my_list_true)
            allocation_mask.extend(my_list_False)
        return allocation_mask
print(convert())