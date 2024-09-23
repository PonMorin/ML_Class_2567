
def likelyhood(x,mean,sigma):
    return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))

def posterior(X,X_train_class,mean_,std_):
    product=np.prod(likelyhood(X,mean_,std_),axis=1)
    product=product*(X_train_class.shape[0]/X_train.shape[0])
    return product

p_1=posterior(X_test,class_data_dic[1],mean_1,std_1)   
p_0=posterior(X_test,class_data_dic[0],mean_0,std_0)
y_pred=1*(p_1>p_0)