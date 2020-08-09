a = 'C:\\Users\\Jason\\Documents\\GitHub\\chesspdftofen\\data\\out\\yasser\\WhiteQueen'
l = []
for e in sorted(os.listdir(a)):
  i = cv2.imread(os.path.join(a, e), 0)
  h,w = i.shape
  # l.append([np.sum(i[h//4:h//4*3, w//4:w//4*3]), e])
  l.append([np.sum(i), e])

l.sort()
print(l)

for i, k in enumerate(l):
  # aa = k[1].find('_')
  # bb = k[1][aa:]
  # os.rename(os.path.join(a, k[1]), os.path.join(a, str(i) + bb))
  os.rename(os.path.join(a, k[1]), os.path.join(a, ('%06d_' % (i,)) + '_' + k[1]))