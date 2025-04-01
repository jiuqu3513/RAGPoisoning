def get_line_end_positions(doc):
    positions = [i for i, char in enumerate(doc) if char == '\n']
    # 处理最后一行无换行符的情况
    if doc and doc[-1] != '\n':
        positions.append(len(doc))
    return positions

# 示例用法
document = """Line1\n
Line2\nLine3"""
line_end_positions = get_line_end_positions(document)
print(line_end_positions)  # 输出: [5, 11, 17]