import numpy as np
import subprocess as sp
from pathlib import Path
import shutil
import glob
from scipy.io import mmread, mmwrite

def vecread(fname):
    return np.array([np.double(line.rstrip()) for line in open(fname, "r")])

def split_matrix(A, splits):
    m, n = A.shape
    s = n // splits
    for i in range(splits):
        yield A[:,i*s:(i+1)*s]

if __name__ == "__main__":

    p = 10
    proc_counts = [2, 4, 8]

    tmpdir = Path("tmpdir")
    tmpdir.mkdir(exist_ok=True)

    for Aname in glob.glob("testdata/*_A.mtx"):
        prefix = Aname.split("/")[-1].split("_A.mtx")[0]
        A = mmread(Aname)
        for nprocs in proc_counts:
            for i, Ai in enumerate(split_matrix(A, nprocs)):
                outpath = tmpdir.joinpath(f"{prefix}_A_{i+1}_{nprocs}.mtx")
                if outpath.is_file():
                    outpath.unlink()
                mmwrite(str(outpath), Ai)
            iprefix = str(tmpdir.joinpath(prefix))
            oprefix = str(Path("testdata").joinpath(prefix))
            proc = sp.Popen(["mpirun", "-np", str(nprocs), "./build/svd_mpi/svd_mpi_bench", "stored", str(p), iprefix, oprefix], stdout=sp.PIPE, stderr=sp.PIPE)
            proc.wait()

            if proc.returncode != 0: continue

            Utest = mmread(f"{oprefix}_Utest.mtx")
            Vttest = mmread(f"{oprefix}_Vttest.mtx")
            Stest = vecread(f"{oprefix}_Stest.txt")
            Up = mmread(f"{oprefix}_U.mtx")
            Vtp = mmread(f"{oprefix}_Vt.mtx")
            Sp = vecread(f"{oprefix}_S.txt")

            Utest = Utest[:,:p]
            Vttest = Vttest[:p,:]
            Stest = Stest[:p]
            Up = Up[:,:p]
            Vtp = Vtp[:p,:]
            Sp = Sp[:p]

            print(f"[A={Aname}][Uerr={np.linalg.norm(Utest@Utest.T-Up@Up.T):.5e},Verr={np.linalg.norm(Vttest.T@Vttest-Vtp.T@Vtp):.5e},Serr={np.linalg.norm(Stest-Sp):.5e}]")
