def encode(left=None, right=None, dynamic=None):
    if dynamic is not None:
        return str(dynamic)  # For dynamic gestures, simply return the number

    # Handle static gestures for left and right hands
    left_code = f"{left+1:02}" if left is not None else ""
    right_code = f"{right+15:02}" if right is not None else ""
    
    if left_code=='' and right_code =='' and dynamic is None:
        return -1

    return str(left_code + right_code)