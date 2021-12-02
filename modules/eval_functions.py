import numpy as np
import h5py
from tqdm import tqdm, tqdm_notebook
import torch
import faiss

def save_submission(
    embed_qry_d,
    embed_ref_d,
    save_path='./submission.h5',
):

    with h5py.File(save_path, "w") as f:
        f.create_dataset("query", data=embed_qry_d['embedding'])
        f.create_dataset("reference", data=embed_ref_d['embedding'])
        f.create_dataset('query_ids', data=[str(s) for s in embed_qry_d['sample_id']])
        f.create_dataset('reference_ids', data=[str(s) for s in embed_ref_d['sample_id']])
    
    print(f' - Saved: {save_path}')
    return None


def read_submission(
    save_path='./sub.h5',
    ):
    embed_qry_d = {}
    embed_ref_d = {}
    
    with h5py.File(save_path, "r") as f:
        embed_qry_d['embedding'] = f['query'][:]
        embed_ref_d['embedding'] = f['reference'][:]
        embed_qry_d['sample_id'] = f['query_ids'][:].astype(str)
        embed_ref_d['sample_id'] = f['reference_ids'][:].astype(str)
        
    return embed_qry_d, embed_ref_d


def calc_embed_d(
    model,
    dataloader,
    do_simple_augmentation=False,
    device='cuda:0',
):
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        image_ids_v = []
        embeddings_v = []

        for data in tqdm(dataloader):
            img = data['img'].to(device)

            embed_v = model.predict_embedding(
                img,
                do_simple_augmentation=do_simple_augmentation,
                calc_triplet_cls=False
            ).detach().cpu().numpy()

            image_ids_v.append( data['sample_id'] )
            embeddings_v.append( embed_v )


    image_ids_v = np.concatenate(image_ids_v, axis=0)
    embeddings_v = np.concatenate(embeddings_v, axis=0)


    embed_d = {
        'embedding':embeddings_v,
        'sample_id':image_ids_v
    }
    
    return embed_d


def add_GT_description_v3(embed_qry_d, embed_ref_d, public_gt, factor=1.0):
    """
    returns a copy of "embed_qry_d" with the ref embedding replaced.
    it points embed_qry towards embed_ref mantaining L2(embed_qry).
    """
    
    embed_qry_d = {
        'embedding': embed_qry_d['embedding'].copy(),
        'sample_id': embed_qry_d['sample_id'].copy(),
    }

    for query_id, reference_id in tqdm(public_gt[~public_gt.reference_id.isna()].values):
        
        query_idx = int(query_id[1:])
        reference_idx = int(reference_id[1:]) 
        
        norm_qry = np.linalg.norm( embed_qry_d['embedding'][query_idx] )
        embed_qry_d['embedding'][query_idx] = embed_ref_d['embedding'][reference_idx] * norm_qry * factor

    return embed_qry_d


def add_GT_description_v2(embed_qry_d, embed_ref_d, public_gt):
    """
    returns a copy of "embed_qry_d" with the ref embedding replaced.
    it points embed_qry towards embed_ref mantaining L2(embed_qry).
    """
    
    embed_qry_d = {
        'embedding': embed_qry_d['embedding'].copy(),
        'sample_id': embed_qry_d['sample_id'].copy(),
    }

    for query_id, reference_id in tqdm(public_gt[~public_gt.reference_id.isna()].values):
        
        query_idx = int(query_id[1:])
        reference_idx = int(reference_id[1:]) 
        
        norm_qry = np.linalg.norm( embed_qry_d['embedding'][query_idx] )
        embed_qry_d['embedding'][query_idx] = embed_ref_d['embedding'][reference_idx] * norm_qry

    return embed_qry_d


def add_GT_description(embed_qry_d, embed_ref_d, public_gt):
    """
    returns a copy of "embed_qry_d" with the ref embedding replaced.
    """
    
    embed_qry_d = {
        'embedding': embed_qry_d['embedding'].copy(),
        'sample_id': embed_qry_d['sample_id'].copy(),
    }

    for query_id, reference_id in tqdm(public_gt[~public_gt.reference_id.isna()].values):
        
        query_idx = int(query_id[1:])
        reference_idx = int(reference_id[1:]) 

        embed_qry_d['embedding'][query_idx] = embed_ref_d['embedding'][reference_idx]

    return embed_qry_d




def calc_match_scores(embed_qry_d, embed_ref_d, k=500, steps=100, gpu_id=None):
    
    dim = embed_ref_d['embedding'].shape[1]
    
    assert embed_qry_d['embedding'].shape[1] == embed_ref_d['embedding'].shape[1]
    
    if gpu_id is not None:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu_id
        index = faiss.GpuIndexFlatL2(
            faiss.StandardGpuResources(),
            dim,
            cfg,
        )

    else:
        index = faiss.IndexFlatL2( dim )
        
        
    index.add( embed_ref_d['embedding'] )
    
    n_samples = len(embed_qry_d['embedding'])
    d_steps   = n_samples // steps
    
    D_v = []
    I_v = []
    for i in tqdm( range(steps) ):
        i_s = d_steps * i
        i_e = (i_s + d_steps) if (i < steps-1) else  n_samples
        
        D, I = index.search( embed_qry_d['embedding'][i_s:i_e], k)
        
        D_v.append(D)
        I_v.append(I.astype(np.int32))
    
    D = np.concatenate(D_v)
    I = np.concatenate(I_v)
    
    C = 1.0 - 0.5 * D
    
    match_scores_d = {'I':I.astype(np.int32), 'C':C}
    
    return match_scores_d
    
    

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def get_pca_parameters(pca):
    A  = np.zeros( (pca.d_in * pca.d_out) )
    PCAMat = np.zeros( (pca.d_in * pca.d_out) )

    b  = np.zeros( (pca.d_out) )
    ev = np.zeros( (pca.d_in) )

    # for i in range(pca.d_in):
    #     ev[i] = pca.eigenvalues.at(i)
    #     for j in range(pca.d_out):
    #         A[i,j] = pca.A.at(j + i*pca.d_out )
    #         PCAMat[i,j] = pca.PCAMat.at(j + i*pca.d_out )


    for i in range(pca.d_in*pca.d_out):
        A[i] = pca.A.at(i)
        PCAMat[i] = pca.PCAMat.at(i)

    PCAMat = PCAMat.reshape( (-1,pca.d_in))
    A = A.reshape( (-1,pca.d_in))


    for i in range(pca.d_in):
        ev[i] = pca.eigenvalues.at(i)

    for j in range(pca.d_out):
        b[j] = pca.b.at(j)
    
    ret_d = {
        'A':A,
        'b':b,
        'ev':ev,
        'PCAMat':PCAMat,
        
    }

#     plt.plot(ev)
#     (PCAMat == A).all()
    
    return ret_d
