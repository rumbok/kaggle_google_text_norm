from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


def sparse_memory_usage(mat):
    try:
        if isinstance(mat, csr_matrix):
            return (mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes) / 1024 / 1024
        if isinstance(mat, coo_matrix):
            return (mat.data.nbytes + mat.row.nbytes + mat.col.nbytes) / 1024 / 1024
    except AttributeError:
        return -1


def dump_svmlight_file_sparse(X, y, filename, zero_based=True):
    with open(filename, mode='w+') as f:
        if isinstance(X, csr_matrix) or isinstance(X, csc_matrix):
            for j, lbl in tqdm(enumerate(y), f'dump {filename}', total=len(y)):
                left = X.indptr[j]
                if j+1 < len(X.indptr):
                    right = X.indptr[j+1]
                else:
                    right = len(X.indptr)

                f.write(str(int(lbl)))
                for i in range(left, right):
                    f.write(f' {X.indices[i]}:{X.data[i]}')
                f.write('\n')

        if isinstance(X, coo_matrix):
            print(y)
            print(X)
            print(X.data)
            print(X.row)
            print(X.col)


        # if not zero_based:
        #     X.col += 1
        #
        # y = y.astype(int).astype(str)
        # col = X.col.astype(str)
        # dat = X.data.astype(str)
        #
        # last_row = 0
        # f.write(y[0])
        #
        # for i in tqdm(range(len(X.data)), f'dump {filename}'):
        #     while last_row < X.row[i]:
        #         last_row += 1
        #         f.write('\n'+y[last_row])
        #     f.write(' '+col[i]+':'+dat[i])
        # f.write('\n')