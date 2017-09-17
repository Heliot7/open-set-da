#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include "libMR/MetaRecognition.h"
#include "svm.h"
#include "mex.h"

int libsvm_version = LIBSVM_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n){
	dst = new T[n];
	memcpy((void *)dst,(void *)src,(ulong)sizeof(T)*n);
}



//
// decision_functionz
//
struct decision_function{
	double *alpha;
	double rho;	
};


void openset_analyze_set(const struct svm_problem& prob, struct svm_model *model,   double *alpha_ptr,double *omega_ptr, int correct_label);
void openset_analyze_pairs(const struct svm_problem &prob,  struct svm_model *model);

// using greedy initalization in openset does not always help,  if OPENSET_GREEDINIT we do it, it not we don't
#define OPENSET_GREEDINIT


static inline double powi(double base, int times){
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2){
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc(((unsigned long)(n))*sizeof(type))

#define ALPHA 0.05

static void print_string_stdout(const char *s){
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...){
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// Kernel Cache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class Cache
{
public:
	Cache(int l,long int size);
	~Cache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	
private:
	int l;
	long int size;
	struct head_t{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

Cache::Cache(int l_,long int size_):l(l_),size(size_){
  head = (head_t *)calloc((ulong) l,(ulong)sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= (ulong)l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (long int) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

Cache::~Cache(){
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void Cache::lru_delete(head_t *h){
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void Cache::lru_insert(head_t *h){
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int Cache::get_data(const int index, Qfloat **data, int len){
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0){
		// free old space
		while(size < more){
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,(ulong)sizeof(Qfloat)*(ulong)len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void Cache::swap_index(int i, int j){
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next){
		if(h->len > i){
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// Kernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of Kernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class QMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}
protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const svm_node *px, const svm_node *py);
	double kernel_linear(int i, int j) const{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const{
		return x[i][(int)(x[j][0].value)].value;
	}
};

Kernel::Kernel(int l, svm_node * const * x_, const svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0){
	switch(kernel_type){
		case LINEAR:
			kernel_function = &Kernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &Kernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &Kernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &Kernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &Kernel::kernel_precomputed;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF){
		x_square = new double[l];
		for(int i=0;i<l;i++)
			x_square[i] = dot(x[i],x[i]);
	}
	else
		x_square = 0;
}

Kernel::~Kernel(){
	delete[] x;
	delete[] x_square;
}

double Kernel::dot(const svm_node *px, const svm_node *py){
	double sum = 0;
	while(px->index != -1 && py->index != -1){
		if(px->index == py->index){
			sum += px->value * py->value;
			++px;
			++py;
		}
		else{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}

double Kernel::k_function(const svm_node *x, const svm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(x,y);
		case POLY:
			return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			while(x->index != -1 && y->index !=-1){
				if(x->index == y->index){
					double d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else{
					if(x->index > y->index){	
						sum += y->value * y->value;
						++y;
					}
					else{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1){
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1){
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(x,y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x[(int)(y->value)].value;
		default:
			return 0;  // Unreachable 
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
	};

	void Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

	double get_C(int i){
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i){
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);	
};

void Solver::swap_index(int i, int j){
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void Solver::reconstruct_gradient(){
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWarning: using -h 0 may be faster\n");

	if (nr_free*l > 2*active_size*(l-active_size)){
		for(i=active_size;i<l;i++){
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i] += alpha[j] * Q_i[j];
		}
	}
	else{
		for(i=0;i<active_size;i++)
			if(is_free(i)){
				const Qfloat *Q_i = Q->get_Q(i,l);
				double alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

void Solver::Solve(int l, const QMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking){
	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++){
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i)){
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}
	}

	// optimization step

	int iter = 0;
	int counter = min(l,1000)+1;

	while(1){
		// show progress and do shrinking

		if(--counter == 0){
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(select_working_set(i,j)!=0){
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*");
			if(select_working_set(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		
		++iter;

		// update alpha[i] and alpha[j], handle bounds carefully
		
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j]){
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0){
				if(alpha[j] < 0){
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else{
				if(alpha[i] < 0){
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j){
				if(alpha[i] > C_i){
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else{
				if(alpha[j] > C_j){
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i){
				if(alpha[i] > C_i){
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else{
				if(alpha[j] < 0){
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j){
				if(alpha[j] > C_j){
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else{
				if(alpha[i] < 0){
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<active_size;k++){
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i)){
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j)){
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
}

// return 1 if already optimal, return 0 otherwise
int Solver::select_working_set(int &out_i, int &out_j){
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax){
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i,active_size);

	for(int j=0;j<active_size;j++){
		if(y[j]==+1){
			if (!is_lower_bound(j)){
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0){
					double obj_diff; 
					double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min){
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else{
			if (!is_upper_bound(j)){
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0){
					double obj_diff; 
					double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min){
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool Solver::be_shrunk(int i, double Gmax1, double Gmax2){
	if(is_upper_bound(i)){
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i)){
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

void Solver::do_shrinking(){
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++){
		if(y[i]==+1)	{
			if(!is_upper_bound(i))	{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	{
			if(!is_upper_bound(i))	{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) {
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2)){
			active_size--;
			while (active_size > i){
				if (!be_shrunk(active_size, Gmax1, Gmax2)){
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver::calculate_rho(){
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++){
		double yG = y[i]*G[i];

		if(is_upper_bound(i)){
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i)){
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}

//
// Solver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class Solver_NU : public Solver
{
public:
	Solver_NU() {}
	void Solve(int l, const QMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking){
		this->si = si;
		Solver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int Solver_NU::select_working_set(int &out_i, int &out_j){
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1){
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp){
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn){
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++){
		if(y[j]==+1){
			if (!is_lower_bound(j))	{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0){
					double obj_diff; 
					double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min){
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else{
			if (!is_upper_bound(j)){
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0){
					double obj_diff; 
					double quad_coef = QD[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool Solver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4){
	if(is_upper_bound(i)){
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else	
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i)){
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void Solver_NU::do_shrinking()
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++){
		if(!is_upper_bound(i)){
			if(y[i]==+1){
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i)){
			if(y[i]==+1){
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) {
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4)){
			active_size--;
			while (active_size > i){
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4)){
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double Solver_NU::calculate_rho()
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<active_size;i++){
		if(y[i]==+1){
			if(is_upper_bound(i))
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
			else{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;
	
	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;
	
	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class SVC_Q: public Kernel{ 
public:
	SVC_Q(const svm_problem& prob, const svm_parameter& param, const schar *y_)
	:Kernel(prob.l, prob.x, param){
		clone(y,y_,prob.l);
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len){
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(y[i]*y[j]*(this->*kernel_function)(i,j));
		}
		return data;
	}

	double *get_QD() const{
		return QD;
	}

	void swap_index(int i, int j) const{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~SVC_Q(){
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	Cache *cache;
	double *QD;
};

class ONE_CLASS_Q: public Kernel{
public:
	ONE_CLASS_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param){
		cache = new Cache(prob.l,(long int)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			for(j=start;j<len;j++)
				data[j] = (Qfloat)(this->*kernel_function)(i,j);
		}
		return data;
	}

	double *get_QD() const{
		return QD;
	}

	void swap_index(int i, int j) const{
		cache->swap_index(i,j);
		Kernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q(){
		delete cache;
		delete[] QD;
	}
private:
	Cache *cache;
	double *QD;
};

class SVR_Q: public Kernel{ 
public:
	SVR_Q(const svm_problem& prob, const svm_parameter& param)
	:Kernel(prob.l, prob.x, param){
		l = prob.l;
		cache = new Cache(l,(long int)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++){
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = (this->*kernel_function)(k,k);
			QD[k+l] = QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}
	
	Qfloat *get_Q(int i, int len) const{
		Qfloat *data;
		int j, real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l){
			for(j=0;j<l;j++)
				data[j] = (Qfloat)(this->*kernel_function)(real_i,j);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
	}

	double *get_QD() const{
		return QD;
	}

	~SVR_Q(){
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	Cache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

//
// construct and solve various formulations
//
static void solve_c_svc(
	const svm_problem *prob, const svm_parameter* param,
	double *alpha, Solver::SolutionInfo* si, double Cp, double Cn){
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;

	for(i=0;i<l;i++){
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	Solver s;
	s.Solve(l, SVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void solve_nu_svc(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si){
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1){
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = new double[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	Solver_NU s;
	s.Solve(l, SVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}
static void solve_one_class(
                            const svm_problem *prob, const svm_parameter *param,
                            double *alpha, Solver::SolutionInfo* si){
	int l = prob->l;
	double *zeros = new double[l];
	schar *ones = new schar[l];
	int i;
    
	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound
    
	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;
    
	for(i=0;i<l;i++){
		zeros[i] = 0;
		ones[i] = 1;
	}
    
	Solver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
            alpha, 1.0, 1.0, param->eps, si, param->shrinking);
    
	delete[] zeros;
	delete[] ones;
}

static void solve_one_class_wsvm(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si){
	int len = prob->l;
	int i;
        int poslabel=1; // *fixme should get label from param  so can do multi-class form 

        struct svm_problem subprob;
        subprob.l = len = 0;
	for(i=0;i<prob->l;i++){
          if(prob->y[i] == poslabel)  {
            len++;
          }
        }
        /* malloc for positive-only subclass */ 
        subprob.x = Malloc(struct svm_node*,len);
        subprob.y = Malloc(double,len);
        subprob.l = len;
        len=0;
	for(i=0;i<prob->l;i++){
          if(prob->y[i] == poslabel)  {
            subprob.x[len] = prob->x[i];
            subprob.y[len] = prob->y[i];
            len++;
          }
        }
        subprob.l = len;
	double *zeros = new double[len];
	schar *ones = new schar[len];


        
	int n = (int)(param->nu*subprob.l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<subprob.l)
           alpha[n] = param->nu * subprob.l - n;
           for(i=n+1;i<prob->l;i++) /* use full size for reseting alspha */ 
		alpha[i] = 0;

	for(i=0;i<len;i++){
		zeros[i] = 0;
		ones[i] = 1;
	}

	Solver s;
	s.Solve(len, ONE_CLASS_Q(subprob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

        free(subprob.x);
        free(subprob.y); 
	delete[] zeros;
	delete[] ones;
}

static void solve_epsilon_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si){
	int l = prob->l;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	for(i=0;i<l;i++){
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	Solver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++){
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void solve_nu_svr(
	const svm_problem *prob, const svm_parameter *param,
	double *alpha, Solver::SolutionInfo* si){
	int l = prob->l;
	double C = param->C;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++){
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	Solver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}


static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input){
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL){
		max_line_len *= 2;
		line = (char *) realloc(line,(ulong)max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


struct svm_model* init_svm_model(const struct svm_parameter * param){
  struct svm_model* model = Malloc(svm_model,1);
  memset(model,0,sizeof(struct svm_model));
  model->param = *param;
  model->rho = NULL;
  model->probA = NULL;
  model->probB = NULL;
  model->label = NULL;
  model->nSV = NULL;
  model->alpha = NULL;
  model->omega = NULL;
  model->openset_dim = 0;
  model->param.rejectedID = -9999; /* id for rejected classes (-99999 is the default) */ 
  model->free_sv = 0;	// XXX
  return model;
}


static decision_function svm_train_one(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn){
	double *alpha = Malloc(double,prob->l);
	Solver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
		case ONE_VS_REST_WSVM:
        case PI_SVM:
		case OPENSET_BIN:
			solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
            solve_one_class(prob,param,alpha,&si);
            break;
		case ONE_WSVM:
		case OPENSET_OC:
			solve_one_class_wsvm(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++){
		if(fabs(alpha[i]) > 0){
			++nSV;
			if(prob->y[i] > 0){
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels, 
	double& A, double& B){
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter; 
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++){
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++){
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++){
			fApB = dec_values[i]*A+B;
			if (fApB >= 0){
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step){
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++){
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B){
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p){
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;
	
	for (t=0;t<k;t++){
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++){
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++){
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++){
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++){
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++){
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++){
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++){
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void svm_binary_svc_probability(
	const svm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB){
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++){
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++){
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct svm_problem subprob;

		subprob.l = prob->l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++){
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++){
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct svm_model *submodel = svm_train(&subprob,&subparam);
			for(j=begin;j<end;j++){
				svm_predict_values(submodel,prob->x[perm[j]],&(dec_values[perm[j]])); 
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}		
			svm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}		
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double svm_svr_probability(
	const svm_problem *prob, const svm_parameter *param){
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	svm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}		
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std) 
			count=count+1;
		else 
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm){
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);	
	int i;

	for(i=0;i<l;i++){
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++){
			if(this_label == label[j]){
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class){
			if(nr_class == max_nr_class){
				max_nr_class *= 2;
				label = (int *)realloc(label,(ulong)max_nr_class*sizeof(int));
				count = (int *)realloc(count,(ulong)max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++){
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}



svm_model *svm_train_oneclass_or_regression(svm_model* model, const svm_problem *prob, const svm_parameter *param){
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR)){
			model->probA = Malloc(double,1);
			model->probA[0] = svm_svr_probability(prob,param);
		}

		decision_function f = svm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(svm_node *,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0){
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				++j;
			}		

		free(f.alpha);
                return model;
}

svm_model *svm_train_binary_pairs(svm_model* model, const svm_problem *prob, const svm_parameter *param){
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
		svm_node **x = Malloc(svm_node *,l);
		int i;
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++){	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,(nr_class*(nr_class-1)/2));

		double *probA=NULL,*probB=NULL;
		if (param->probability){
                  probA=Malloc(double,(nr_class*(nr_class-1)/2));
                  probB=Malloc(double,(nr_class*(nr_class-1)/2));
		}

		int p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				svm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(svm_node *,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++){
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++){
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}

				if(param->probability)
					svm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

				f[p] = svm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;
		
		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(double,(nr_class*(nr_class-1)/2));
		for(i=0;i<(nr_class*(nr_class-1)/2);i++)
			model->rho[i] = f[i].rho;

		if(param->probability){
			model->probA = Malloc(double,(nr_class*(nr_class-1)/2));
			model->probB = Malloc(double,(nr_class*(nr_class-1)/2));
			for(i=0;i<(nr_class*(nr_class-1)/2);i++){
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++){
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j]){	
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}
		
		info("Total nSV = %d\n",total_sv);

		model->l = total_sv;
		model->SV = Malloc(svm_node *,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i]) model->SV[p++] = x[i];

		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,(nr_class-1));
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++){
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];
				
				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}
		
		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<(nr_class*(nr_class-1)/2);i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
                return model;
}

svm_model *svm_train_onevset_oneclass(svm_model* model, const svm_problem *prob, const svm_parameter *param){

  int nr_class,i;
  int *label = NULL;
  int *start = NULL;
  int *count = NULL;
  int l = prob->l;
  int *perm = Malloc(int,l);
  
  // group training data of the same class
  svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
  
  /* save a copy of orginal data */ 
  svm_node **x = Malloc(svm_node *,l);
  for(i=0;i<l;i++){
    x[i] = prob->x[perm[i]];
  }

  model->nr_class = nr_class;

  /* we make an array of temp models to use during fitting, then we merge into one multi-class model */ 
  struct svm_model  **tmodels;
  tmodels = Malloc(svm_model*,nr_class+1);
  
  struct svm_parameter tparam;
  memcpy(&tparam,param,sizeof(tparam));
  tparam.svm_type = ONE_CLASS;
  tparam.do_open=1;
  int keep_cnt=0;
  for(i=0;i<nr_class;i++){
    if((model->param.neg_labels == false && label[i] < 0)) {
      /* do we allow negative classes */ 
      tmodels[i] =  NULL;
    }else {
      keep_cnt++;
      tmodels[i] =  Malloc(svm_model,1);
      memset(tmodels[i],0,sizeof(struct svm_model));
      tmodels[i]->openset_dim = 1;
      tmodels[i]->rho =   tmodels[i]->alpha =   tmodels[i]->omega = 0;
      tmodels[i]->label=0;
      tmodels[i]->param = tparam;
      tmodels[i]->free_sv = 0;	// we will allocate space, so mark it to be cleaned up
    }
  }
  
  model->param.svm_type=OPENSET_OC;
  /* setup model for multiple openset classes */
  model->openset_dim = nr_class;
  model->openset_dim = keep_cnt;
  model->rho = Malloc(double,model->openset_dim);                
  model->alpha = Malloc(double,model->openset_dim);                
  model->omega = Malloc(double,model->openset_dim);                
  model->label = Malloc(int,nr_class);


  /* loop over our classes, pull out data, make a model,, train on it, then openset analysze it. */
  for(i=0;i<nr_class;i++){
      model->label[i] = label[i];
      if(tmodels[i] == NULL) continue; 
      svm_problem sub_prob;
      int si = start[i];
      int ci = count[i];
      sub_prob.l = prob->l;
      sub_prob.x = Malloc(svm_node *,sub_prob.l);
      sub_prob.y = Malloc(double,sub_prob.l);
      int k;
      for(k=0;k<ci; k++){
          sub_prob.x[k] = x[si+k];
          sub_prob.y[k] = +1;
        }
      for(k=0;k<si;k++){
          sub_prob.x[ci+k] = x[k];
          sub_prob.y[ci+k] = -1;
        }
      for(k=ci+si;k<sub_prob.l;k++){
          sub_prob.x[k] = x[k];
          sub_prob.y[k] = -1;
        }

      tmodels[i] = svm_train_oneclass_or_regression(tmodels[i],&sub_prob,param);
      model->rho[i] = tmodels[i]->rho[0];
      /* now analyze subproblem for openset adjustments, saving resulting alpha and omega  */ 
      openset_analyze_set(sub_prob,  tmodels[i], &model->alpha[i],&model->omega[i],+1);
      
                   
      free(sub_prob.x);
      free(sub_prob.y);
    }
  
  if(param->probability){
    info("Probability not yet supported for openset");
    //put prob code hwere when it is
  }
  else{
      model->probA=NULL;
      model->probB=NULL;
    }
  
  /* now we allocate overall SV for the set, and copy data back into the multi-class model */ 
  int total_sv = 0;
  keep_cnt=0;
  for(i=0;i<nr_class;i++) {
    if(tmodels[i])  {
      total_sv +=  tmodels[i]->l;
      model->alpha[keep_cnt] = model->alpha[i];
      model->omega[keep_cnt] = model->omega[i];
      //      model->label[keep_cnt] = model->label[i];
      model->rho[keep_cnt] = model->rho[i];
      keep_cnt++;
    } else {
      //      model->label[nr_class-(i-keep_cnt)-1] = model->label[i];// collect negative labels at the end.. there are no models 
    }
  }
  info("Total nSV = %d\n",total_sv);

  model->openset_dim = keep_cnt;
  model->l = total_sv;
  int svcnt = 0; /* where are we on overall SV count */


  if(keep_cnt == nr_class){
    model->nSV = Malloc(int ,model->nr_class+1);
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,(model->nr_class+1));
    for(i=0;i<nr_class;i++){
      //    fprintf(stderr,"Copying over class %d with %d support vectors\n",i,tmodels[i]->l);
      if(tmodels[i]) {    
        model->nSV[i] = (int)tmodels[i]->l;
        model->sv_coef[i] = Malloc(double,total_sv);
        memset(model->sv_coef[i],0,total_sv*sizeof(double));
        for(int k=0;k<model->nSV[i];k++,svcnt++){
          model->SV[svcnt] = tmodels[i]->SV[k];
          model->sv_coef[i][svcnt] = tmodels[i]->sv_coef[0][k];
        }
      }
    }

  for(i=0;i<nr_class;i++){
    if(tmodels[i]) {
      svm_free_model_content(tmodels[i]);
      free(tmodels[i]);
    }
  }

  } else {
    /* if we have negative labesl to be ignored, we need to save the models a little differnt */   
    if(!(keep_cnt == 1 && nr_class==2)){
      fprintf(stderr,"Openset ignoring  negative classess currently only supported for binary.. did you forget the -B flag?\n"
              "This may not  build a valid model and will leak memory");
    }

    int keepit=0;
    if(tmodels[1]) { 
      keepit=1;
    }
    model->rho[0] = tmodels[keepit]->rho[0];
    model->SV = tmodels[keepit]->SV;
    model->nSV = tmodels[keepit]->nSV;
    model->sv_coef =   tmodels[keepit]->sv_coef;

    for(i=0;i<nr_class;i++){
      if(tmodels[i]) {
        //      svm_free_model_content(tmodels[i]);// we reused some of the content, don't free them all,do only those we don't need
        free(tmodels[i]->label);
        free(tmodels[i]->rho);
        free(tmodels[i]);
      }
    }
  }

  model->free_sv=0;

  free(tmodels);

  free(x);
  free(label);
  free(start);
  free(count);
  free(perm);

  return model;
}

// .. One_class_wsvm
svm_model *svm_train_wsvm_onevset_oneclass(svm_model* model, const svm_problem *prob, const svm_parameter *param){

  int nr_class,i;
  int *label = NULL;
  int *start = NULL;
  int *count = NULL;
  int l = prob->l;
  int *perm = Malloc(int,l);
  // group training data of the same class
  svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
  /* save a copy of orginal data */ 
  svm_node **x = Malloc(svm_node *,l);
  for(i=0;i<l;i++){
    x[i] = prob->x[perm[i]];
  }
  model->nr_class = nr_class;
  /* we make an array of temp models to use during fitting, then we merge into one multi-class model */ 
  struct svm_model  **tmodels;
  tmodels = Malloc(svm_model*,nr_class+1);
  struct svm_parameter tparam;
  memcpy(&tparam,param,sizeof(tparam));
  tparam.svm_type = ONE_CLASS;
  tparam.do_open=1;
  int keep_cnt=0;
  for(i=0;i<nr_class;i++){
   if((model->param.neg_labels == false && label[i] < 0)) {
      /* do we allow negative classes */ 
      tmodels[i] =  NULL;
    }else {
      keep_cnt++;
      tmodels[i] =  Malloc(svm_model,1);
      memset(tmodels[i],0,sizeof(struct svm_model));
      tmodels[i]->openset_dim = 1;
      tmodels[i]->rho =   tmodels[i]->alpha =   tmodels[i]->omega = 0;
      tmodels[i]->label=0;
      tmodels[i]->param = tparam;
      tmodels[i]->free_sv = 0;	// we will allocate space, so mark it to be cleaned up
    }
  }

  /* setup model for multiple openset classes */
  model->openset_dim = nr_class;
  model->openset_dim = keep_cnt;
  model->rho = Malloc(double,model->openset_dim);
  model->label = Malloc(int,nr_class);

  model->MRpos_one_class = new MetaRecognition[model->openset_dim];	
  int weibull_model_count = 0;	
  /* loop over our classes, pull out data, make a model,, train on it, then openset analysze it. */
  for(i=0;i<nr_class;i++){
      model->label[i] = label[i];
      if(tmodels[i] == NULL) continue; 
      svm_problem sub_prob;
      int si = start[i];
      int ci = count[i];
      sub_prob.l = prob->l;
      sub_prob.x = Malloc(svm_node *,sub_prob.l);
      sub_prob.y = Malloc(double,sub_prob.l);
      int k;
      int pcnt=0;	
      for(k=0;k<ci; k++){
          sub_prob.x[k] = x[si+k];
          sub_prob.y[k] = +1;
          pcnt++;
        }
      for(k=0;k<si;k++){
          sub_prob.x[ci+k] = x[k];
          sub_prob.y[ci+k] = -1;
        }
      for(k=ci+si;k<sub_prob.l;k++){
          sub_prob.x[k] = x[k];
          sub_prob.y[k] = -1;
        }

      fprintf(stderr,"Building one-class model with %d pos for class %d\n",pcnt,label[i]);
      tmodels[i] = svm_train_oneclass_or_regression(tmodels[i],&sub_prob,param);
      model->rho[i] = tmodels[i]->rho[0];

      int top_score_pos = (int) ( (double)1.5 * tmodels[i]->l );
      if(top_score_pos >= pcnt)
          top_score_pos = tmodels[i]->l;
      if(top_score_pos < 3){
          fprintf(stderr,"Error: Minimum 3 samples are needed for Weibull fitting in class %d\n",label[i]);
          exit(1);
      }
      /*fprintf(stderr,"+ve top scores %d\n",top_score_pos);
      fprintf(stderr,"total samples %d\n",sub_prob.l);*/
      svm_node_libsvm *sub_nd = Malloc(svm_node_libsvm ,sub_prob.l);
      double *dec_values = (double*)malloc( 1 * sizeof (double));//memory creating for storing decision scores in one-dimensional array	
      for(int q=0;q<sub_prob.l;q++){
      	sub_nd[q].index = sub_prob.y[q];
        svm_predict_values(tmodels[i],sub_prob.x[q],dec_values);
        sub_nd[q].value = dec_values[0];
      }
      model->MRpos_one_class[weibull_model_count].FitSVM(sub_nd, sub_prob.l, +1, true, MetaRecognition::positive_model,top_score_pos );//positive tail w.r.t. positive class 
      if(0) fprintf(stderr,"%lf %lf %lf %lf %lf %lf\n", 
                    model->MRpos_one_class[weibull_model_count].W_score(2.0), 
                    model->MRpos_one_class[weibull_model_count].W_score(1),
                    model->MRpos_one_class[weibull_model_count].W_score(.1),
                    model->MRpos_one_class[weibull_model_count].W_score(-.1),
                    model->MRpos_one_class[weibull_model_count].W_score(-1),
                    model->MRpos_one_class[weibull_model_count].W_score(-2));
      weibull_model_count++;      
      free(sub_prob.x);
      free(sub_prob.y);
    }
  /* now we allocate overall SV for the set, and copy data back into the multi-class model */ 
  int total_sv = 0;
  keep_cnt=0;
  for(i=0;i<nr_class;i++) {
    if(tmodels[i])  {
      total_sv +=  tmodels[i]->l;
      model->rho[keep_cnt] = model->rho[i];
      keep_cnt++;
    }
  }
  info("Total nSV = %d\n",total_sv);

  model->openset_dim = keep_cnt;
  model->l = total_sv;
  int svcnt = 0; /* where are we on overall SV count */
  if(keep_cnt == nr_class){
    model->nSV = Malloc(int ,model->nr_class+1);
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,(model->nr_class+1));
    for(i=0;i<nr_class;i++){
      //    fprintf(stderr,"Copying over class %d with %d support vectors\n",i,tmodels[i]->l);
      if(tmodels[i]) {    
        model->nSV[i] = (int)tmodels[i]->l;
        model->sv_coef[i] = Malloc(double,total_sv);
        memset(model->sv_coef[i],0,total_sv*sizeof(double));
        for(int k=0;k<model->nSV[i];k++,svcnt++){
          model->SV[svcnt] = tmodels[i]->SV[k];
          model->sv_coef[i][svcnt] = tmodels[i]->sv_coef[0][k];
        }
      }
    }

  for(i=0;i<nr_class;i++){
    if(tmodels[i]) {
      svm_free_model_content(tmodels[i]);
      free(tmodels[i]);
    }
  }

  } else {
    /* if we have negative labesl to be ignored, we need to save the models a little differnt */   
    if(!(keep_cnt == 1 && nr_class==2)){
      fprintf(stderr,"Openset ignoring  negative classess currently only supported for binary.. did you forget the -B flag?\n"
              "This may not  build a valid model and will leak memory");
    }

    int keepit=0;
    if(tmodels[1]) { 
      keepit=1;
    }
    model->rho[0] = tmodels[keepit]->rho[0];
    model->SV = tmodels[keepit]->SV;
    model->nSV = tmodels[keepit]->nSV;
    model->sv_coef =   tmodels[keepit]->sv_coef;

    for(i=0;i<nr_class;i++){
      if(tmodels[i]) {
        //      svm_free_model_content(tmodels[i]);// we reused some of the content, don't free them all,do only those we don't need
        free(tmodels[i]->label);
        free(tmodels[i]->rho);
        free(tmodels[i]);
      }
    }
  }
  model->free_sv=0;
  free(tmodels);
  free(x);
  free(label);
  free(start);
  free(count);
  free(perm);

  return model;
}

// .. One_vs_rest_wsvm
svm_model *svm_train_wsvm_onevset_binary(svm_model* model, const svm_problem *prob, const svm_parameter *param){

  int nr_class,i;
  int *label = NULL;
  int *start = NULL;
  int *count = NULL;
  int l = prob->l;
  int *perm = Malloc(int,l);
  // group training data of the same class
  svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
  /* save a copy of orginal data */ 
  svm_node **x = Malloc(svm_node *,l);
  for(i=0;i<l;i++){
    x[i] = prob->x[perm[i]];
  }
  model->nr_class = nr_class;
  /* we make an array of temp models to use during fitting, then we merge into one multi-class model */ 
  struct svm_model  **tmodels;
  tmodels = Malloc(svm_model*,nr_class+1);
  struct svm_parameter tparam;
  memcpy(&tparam,param,sizeof(tparam));
  tparam.svm_type = C_SVC;
  tparam.do_open=1;
  int keep_cnt=0;
  for(i=0;i<nr_class;i++){
    if(((model->param.neg_labels == false && label[i] < 0))) {
      tmodels[i] =  NULL;
    }
    else {
      keep_cnt++;
      tmodels[i] =  Malloc(svm_model,1);
      memset(tmodels[i],0,sizeof(struct svm_model));
      tmodels[i]->openset_dim = 1;
      tmodels[i]->rho = 0;//   tmodels[i]->alpha =   tmodels[i]->omega = 0;
      tmodels[i]->label=0;
      tmodels[i]->param = tparam;
      tmodels[i]->free_sv = 0;	// we will allocate space, so mark it to be cleaned up
	}	
  }
  model->param.svm_type=ONE_VS_REST_WSVM;
    /* setup model for multiple openset classes */
  model->openset_dim = keep_cnt;
  model->MRpos_one_vs_all = new MetaRecognition[model->openset_dim];
  model->MRcomp_one_vs_all = new MetaRecognition[model->openset_dim];	
  model->rho = Malloc(double,model->openset_dim);                
  model->label = Malloc(int,nr_class);
  /* loop over our classes, pull out data, make a model,, train on it, then openset analysze it. */
  int weibull_model_count=0;
  for(i=0;i<nr_class;i++){
      model->label[i] = label[i];
      if(tmodels[i] == NULL) continue; 
      svm_problem sub_prob;
      sub_prob.nr_classes=2;
      int si = start[i];
      int ci = count[i];
      sub_prob.l = prob->l;
      sub_prob.x = Malloc(svm_node *,sub_prob.l);
      sub_prob.y = Malloc(double,sub_prob.l);
      int k;
      int plabel = label[i];
      int nlabel = -label[i];
      if(plabel==nlabel) nlabel-=1;
      int pcnt=0;
      int ncnt=0;
      /* positive class is the first label see, so order it so all positves are at front */ 
      for(k=0;k<ci; k++){
          sub_prob.x[k] = x[si+k];
          sub_prob.y[k] = plabel;
          pcnt++;
      }
      for(k=0;k<si;k++){
          sub_prob.x[ci+k] = x[k];
          sub_prob.y[ci+k] = nlabel;
          ncnt++;
      }
      for(k=ci+si;k<sub_prob.l;k++){
          sub_prob.x[k] = x[k];
          sub_prob.y[k] = nlabel;
          ncnt++;
      }      
      // fprintf(stderr,"Training binary 1-vs-rest WSVM for class %d with %d pos and %d neg examples\n",plabel,pcnt,ncnt);      
	  // mexPrintf("Training binary 1-vs-rest WSVM for class %d with %d pos and %d neg examples\n", plabel, pcnt, ncnt);      
      tmodels[i] = svm_train_binary_pairs(tmodels[i],&sub_prob,param);
      model->rho[i] = tmodels[i]->rho[0];
      /* now analyze subproblem for openset adjustments, saving resulting alpha and omega  */ 
      int top_score_pos = (int)(1.5 * tmodels[i]->nSV[0]) ;
      if( (top_score_pos >= pcnt) || (top_score_pos <= 3) )
          top_score_pos = tmodels[i]->nSV[0];
      int top_score_neg = (int) ((1.5 * tmodels[i]->nSV[1])/(nr_class-1) ) ;
      if( (top_score_neg >= ncnt) )
          top_score_neg = tmodels[i]->nSV[1]/ (nr_class-1);
      if( (top_score_neg <= 3) )
          top_score_neg = tmodels[i]->nSV[1];
      if(top_score_pos < 3){
            fprintf(stderr,"Error: Minimum 3 samples are needed for Weibull fitting in class %d\n",label[i]);
            exit(1);
      }
      if(top_score_neg < 3){
            fprintf(stderr,"Error: Minimum 3 samples are needed for Weibull fitting in negative class of class %d\n",label[i]);
            exit(1);
      }
      /*fprintf(stderr,"+ve top scores %d\n",top_score_pos);
      fprintf(stderr,"-ve top scores %d\n",top_score_neg);
      fprintf(stderr,"total SV's %d\n",tmodels[i]->l);	
      fprintf(stderr,"total samples %d\n",sub_prob.l);*/
      svm_node_libsvm *sub_nd = Malloc(svm_node_libsvm ,sub_prob.l);
      double *dec_values = (double*)malloc( (tmodels[i]->nr_class*(tmodels[i]->nr_class-1)/2) * sizeof (double));//memory creating for storing n*(n-1)/2 pairwise decision scores in one-dimensional array
      for(int q=0;q<sub_prob.l;q++){
      	sub_nd[q].index = sub_prob.y[q];
        svm_predict_values(tmodels[i],sub_prob.x[q],dec_values);
        sub_nd[q].value = dec_values[0];
      }	
      int rvalue = model->MRpos_one_vs_all[weibull_model_count].FitSVM(sub_nd, sub_prob.l, plabel, true, MetaRecognition::positive_model,top_score_pos );//positive tail w.r.t. positive class 
	  // mexPrintf("rvalue MRpos_one_vs_all: %d\n", rvalue);
      if(rvalue!=1)
          printf("fit weibull positive %d\n",rvalue);
      if(0) fprintf(stderr,"%lf %lf %lf %lf %lf %lf\n", 
             model->MRpos_one_vs_all[weibull_model_count].W_score(2.0), 
             model->MRpos_one_vs_all[weibull_model_count].W_score(1),
             model->MRpos_one_vs_all[weibull_model_count].W_score(.1),
             model->MRpos_one_vs_all[weibull_model_count].W_score(-.1),
             model->MRpos_one_vs_all[weibull_model_count].W_score(-1),
             model->MRpos_one_vs_all[weibull_model_count].W_score(-2));
	
      rvalue = model->MRcomp_one_vs_all[weibull_model_count].FitSVM(sub_nd,sub_prob.l, plabel, true, MetaRecognition::complement_reject,top_score_neg);//compliment tail w.r.t. positive class
	  // mexPrintf("rvalue MRcomp_one_vs_all: %d\n", rvalue);
      if(rvalue!=1)				
          printf("fit weibull compliment %d\n",rvalue);
      if(0) fprintf(stderr,"%lf %lf %lf %lf %lf %lf\n", 
             model->MRcomp_one_vs_all[weibull_model_count].W_score(2.0), 
             model->MRcomp_one_vs_all[weibull_model_count].W_score(1),
             model->MRcomp_one_vs_all[weibull_model_count].W_score(.1),
             model->MRcomp_one_vs_all[weibull_model_count].W_score(-.1),
             model->MRcomp_one_vs_all[weibull_model_count].W_score(-1),
             model->MRcomp_one_vs_all[weibull_model_count].W_score(-2));
    	//printf("pair_count = %d\n",i);
      weibull_model_count++;            
      free(sub_prob.x);
      free(sub_prob.y);
    }

  /* now we allocate overall SV for the set, and copy data back into the multi-class model */ 
  int total_sv = 0;
  keep_cnt=0;
  for(i=0;i<nr_class;i++){
    if(tmodels[i]){
      total_sv +=  tmodels[i]->l;

      model->rho[keep_cnt] = model->rho[i];
      keep_cnt++;
    } 
  }
  info("Total nSV = %d\n",total_sv);
  model->openset_dim = keep_cnt;
  model->l = total_sv;
  int svcnt = 0; /* where are we on overall SV count */
  if(keep_cnt == nr_class){
    model->nSV = Malloc(int ,model->nr_class+1);
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,(model->nr_class+1));
    for(i=0;i<nr_class;i++){
      if(tmodels[i]) {    
        model->nSV[i] = (int)tmodels[i]->l;
        model->sv_coef[i] = Malloc(double,total_sv);
        memset(model->sv_coef[i],0,total_sv*sizeof(double));
        for(int k=0;k<model->nSV[i];k++,svcnt++){
          model->SV[svcnt] = tmodels[i]->SV[k];
          model->sv_coef[i][svcnt] = tmodels[i]->sv_coef[0][k];
        }
      }
    }

  for(i=0;i<nr_class;i++){
    if(tmodels[i]) {
      svm_free_model_content(tmodels[i]);
      free(tmodels[i]);
    }
  }

  }
  else{
    mexPrintf("ENTERS means more code to cheack :(\n");
    /* if we have negative labesl to be ignored, we need to save the models a little differnt */   
    if(!(keep_cnt == 1 && nr_class==2)){
      fprintf(stderr,"Openset ignoring  negative classess currently only supported for binary.. did you forget the -B flag?\n"
              "This may not  build a valid model and will leak memory");
    }
    int keepit=0;
    if(tmodels[1]) { 
      keepit=1;
    }
    model->rho[0] = tmodels[keepit]->rho[0];
    model->SV = tmodels[keepit]->SV;
    model->nSV = tmodels[keepit]->nSV;
    model->sv_coef =   tmodels[keepit]->sv_coef;

    for(i=0;i<nr_class;i++){
      if(tmodels[i]) {
        //      svm_free_model_content(tmodels[i]);// we reused some of the content, don't free them all,do only those we don't need
        free(tmodels[i]->label);
        free(tmodels[i]->rho);
        free(tmodels[i]);
      }
    }
  }
  model->free_sv=0;
  free(tmodels);
  free(x);
  free(label);
  free(start);
  free(count);
  free(perm);

  return model;
}

svm_model *svm_train_onevset_binary(svm_model* model, const svm_problem *prob, const svm_parameter *param){

  int nr_class,i;
  int *label = NULL;
  int *start = NULL;
  int *count = NULL;
  int l = prob->l;
  int *perm = Malloc(int,l);
  
  // group training data of the same class
  svm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
  
  /* save a copy of orginal data */ 
  svm_node **x = Malloc(svm_node *,l);
  for(i=0;i<l;i++){
    x[i] = prob->x[perm[i]];
  }

  model->nr_class = nr_class;

  /* we make an array of temp models to use during fitting, then we merge into one multi-class model */ 
  struct svm_model  **tmodels;
  tmodels = Malloc(svm_model*,nr_class+1);
  
  struct svm_parameter tparam;
  memcpy(&tparam,param,sizeof(tparam));
  tparam.svm_type = C_SVC;
  tparam.do_open=1;
  int keep_cnt=0;
  for(i=0;i<nr_class;i++){
    if(((model->param.neg_labels == false && label[i] < 0))) {
      tmodels[i] =  NULL;
    } else {
      keep_cnt++;
      tmodels[i] =  Malloc(svm_model,1);
      memset(tmodels[i],0,sizeof(struct svm_model));
      tmodels[i]->openset_dim = 1;
      tmodels[i]->rho =   tmodels[i]->alpha =   tmodels[i]->omega = 0;
      tmodels[i]->label=0;
      tmodels[i]->param = tparam;
      tmodels[i]->free_sv = 0;	// we will allocate space, so mark it to be cleaned up
    }
  }
  
  
  model->param.svm_type=OPENSET_BIN;
    /* setup model for multiple openset classes */
  model->openset_dim = keep_cnt;
  model->rho = Malloc(double,model->openset_dim);                
  model->alpha = Malloc(double,model->openset_dim);                
  model->omega = Malloc(double,model->openset_dim);                
  model->label = Malloc(int,nr_class);
  /* loop over our classes, pull out data, make a model,, train on it, then openset analysze it. */

  for(i=0;i<nr_class;i++){
      model->label[i] = label[i];
      if(tmodels[i] == NULL) continue; 
      svm_problem sub_prob;
      sub_prob.nr_classes=2;
      int si = start[i];
      int ci = count[i];
      sub_prob.l = prob->l;
      sub_prob.x = Malloc(svm_node *,sub_prob.l);
      sub_prob.y = Malloc(double,sub_prob.l);
      int k;
      int plabel = label[i];
      int nlabel = -label[i];
      if(plabel==nlabel) nlabel-=1;
      int pcnt=0;
      int ncnt=0;
      /* positive class is the first label see, so order it so all positves are at front */ 
      for(k=0;k<ci; k++){
          sub_prob.x[k] = x[si+k];
          sub_prob.y[k] = plabel;
          pcnt++;
      }
      for(k=0;k<si;k++){
          sub_prob.x[ci+k] = x[k];
          sub_prob.y[ci+k] = nlabel;
          ncnt++;
      }
      for(k=ci+si;k<sub_prob.l;k++){
          sub_prob.x[k] = x[k];
          sub_prob.y[k] = nlabel;
          ncnt++;
      }
      if(model->param.vfile)          
        fprintf(model->param.vfile,"Trainingg binary 1-vs-set for class %d with %d pos and %d neg examples\n",plabel,pcnt,ncnt);      
      fprintf(stderr,"Training binary 1-vs-set for class %d with %d pos and %d neg examples\n",plabel,pcnt,ncnt);      
      tmodels[i] = svm_train_binary_pairs(tmodels[i],&sub_prob,param);
      model->rho[i] = tmodels[i]->rho[0];
      /* now analyze subproblem for openset adjustments, saving resulting alpha and omega  */ 
      openset_analyze_set(sub_prob,  tmodels[i], &model->alpha[i],&model->omega[i],plabel);
                   
      free(sub_prob.x);
      free(sub_prob.y);
    }
  
  if(param->probability){
    info("Probability not yet supported for openset");
    //put prob code hwere when it is
  }
  else
    {
      model->probA=NULL;
      model->probB=NULL;
    }
  /* now we allocate overall SV for the set, and copy data back into the multi-class model */ 
  int total_sv = 0;
  keep_cnt=0;
  for(i=0;i<nr_class;i++) {
    if(tmodels[i])  {
      total_sv +=  tmodels[i]->l;
      model->alpha[keep_cnt] = model->alpha[i];
      model->omega[keep_cnt] = model->omega[i];
      //      model->label[keep_cnt] = model->label[i];
      model->rho[keep_cnt] = model->rho[i];
      keep_cnt++;
    }
    else {
      //      model->label[nr_class-(i-keep_cnt)-1] = model->label[i];// collect negative labels at the end.. there are no models 
    }
  }
  info("Total nSV = %d\n",total_sv);

  model->openset_dim = keep_cnt;
  model->l = total_sv;
  int svcnt = 0; /* where are we on overall SV count */


  if(keep_cnt == nr_class){
    model->nSV = Malloc(int ,model->nr_class+1);
    model->SV = Malloc(svm_node *,total_sv);
    model->sv_coef = Malloc(double *,(model->nr_class+1));
    for(i=0;i<nr_class;i++){
      //    fprintf(stderr,"Copying over class %d with %d support vectors\n",i,tmodels[i]->l);
      if(tmodels[i]) {    
        model->nSV[i] = (int)tmodels[i]->l;
        model->sv_coef[i] = Malloc(double,total_sv);
        memset(model->sv_coef[i],0,total_sv*sizeof(double));
        for(int k=0;k<model->nSV[i];k++,svcnt++){
          model->SV[svcnt] = tmodels[i]->SV[k];
          model->sv_coef[i][svcnt] = tmodels[i]->sv_coef[0][k];
        }
      }
    }

  for(i=0;i<nr_class;i++){
    if(tmodels[i]) {
      svm_free_model_content(tmodels[i]);
      free(tmodels[i]);
    }
  }

  } else {
    /* if we have negative labesl to be ignored, we need to save the models a little differnt */   
    if(!(keep_cnt == 1 && nr_class==2)){
      fprintf(stderr,"Openset ignoring  negative classess currently only supported for binary.. did you forget the -B flag?\n"
              "This may not  build a valid model and will leak memory");
    }

    int keepit=0;
    if(tmodels[1]) { 
      keepit=1;
    }
    model->rho[0] = tmodels[keepit]->rho[0];
    model->SV = tmodels[keepit]->SV;
    model->nSV = tmodels[keepit]->nSV;
    model->sv_coef =   tmodels[keepit]->sv_coef;

    for(i=0;i<nr_class;i++){
      if(tmodels[i]) {
        //      svm_free_model_content(tmodels[i]);// we reused some of the content, don't free them all,do only those we don't need
        free(tmodels[i]->label);
        free(tmodels[i]->rho);
        free(tmodels[i]);
      }
    }
  }

  model->free_sv=0;

  free(tmodels);

  free(x);
  free(label);
  free(start);
  free(count);
  free(perm);

  return model;
}

svm_model *svm_train_pi_svm_onevset_binary(svm_model* model, const svm_problem *prob, const svm_parameter *param){
    
    int nr_class,i;
    int *label = NULL;
    int *start = NULL;
    int *count = NULL;
    int l = prob->l;
    int *perm = Malloc(int,l);
    // group training data of the same class
    svm_group_classes(prob,&nr_class,&label,&start,&count,perm);
    /* save a copy of orginal data */
    svm_node **x = Malloc(svm_node *,l);
    for(i=0;i<l;i++){
        x[i] = prob->x[perm[i]];
    }
    model->nr_class = nr_class;
    /* we make an array of temp models to use during fitting, then we merge into one multi-class model */
    struct svm_model  **tmodels;
    tmodels = Malloc(svm_model*,nr_class+1);
    struct svm_parameter tparam;
    memcpy(&tparam,param,sizeof(tparam));
    tparam.svm_type = C_SVC;
    tparam.do_open=1;
    int keep_cnt=0;
    for(i=0;i<nr_class;i++){
        if(((model->param.neg_labels == false && label[i] < 0))) {
            tmodels[i] =  NULL;
        } else {
            keep_cnt++;
            tmodels[i] =  Malloc(svm_model,1);
            memset(tmodels[i],0,sizeof(struct svm_model));
            tmodels[i]->openset_dim = 1;
            tmodels[i]->rho =   tmodels[i]->alpha =   tmodels[i]->omega = 0;
            tmodels[i]->label=0;
            tmodels[i]->param = tparam;
            tmodels[i]->free_sv = 0;	// we will allocate space, so mark it to be cleaned up
        }
    }
    model->param.svm_type=PI_SVM;
    /* setup model for multiple openset classes */
    model->openset_dim = keep_cnt;
    model->MRpos_one_vs_all = new MetaRecognition[model->openset_dim];
    model->rho = Malloc(double,model->openset_dim);
    model->label = Malloc(int,nr_class);
    /* loop over our classes, pull out data, make a model,, train on it, then openset analysze it. */
    int weibull_model_count=0;
    for(i=0;i<nr_class;i++)
    {
        model->label[i] = label[i];
        if(tmodels[i] == NULL) continue;
        svm_problem sub_prob;
        sub_prob.nr_classes=2;
        int si = start[i];
        int ci = count[i];
        sub_prob.l = prob->l;
        sub_prob.x = Malloc(svm_node *,sub_prob.l);
        sub_prob.y = Malloc(double,sub_prob.l);
        int k;
        int plabel = label[i];
        int nlabel = -label[i];
        if(plabel==nlabel) nlabel-=1;
        int pcnt=0;
        int ncnt=0;
        /* positive class is the first label see, so order it so all positves are at front */
        for(k=0;k<ci; k++)
        {
            sub_prob.x[k] = x[si+k];
            sub_prob.y[k] = plabel;
            pcnt++;
        }
        for(k=0;k<si;k++)
        {
            sub_prob.x[ci+k] = x[k];
            sub_prob.y[ci+k] = nlabel;
            ncnt++;
        }
        for(k=ci+si;k<sub_prob.l;k++)
        {
            sub_prob.x[k] = x[k];
            sub_prob.y[k] = nlabel;
            ncnt++;
        }
        if(model->param.vfile)
            fprintf(model->param.vfile,"Trainingg binary 1-vs-set for class %d with %d pos and %d neg examples\n",plabel,pcnt,ncnt);
        fprintf(stderr,"Training binary 1-vs-set for class %d with %d pos and %d neg examples\n",plabel,pcnt,ncnt);
        tmodels[i] = svm_train_binary_pairs(tmodels[i],&sub_prob,param);
        model->rho[i] = tmodels[i]->rho[0];
        /* now analyze subproblem for openset adjustments, saving resulting alpha and omega  */
        int top_score_pos = (int)(1.5 * tmodels[i]->nSV[0]) ;
        if( (top_score_pos >= pcnt) || (top_score_pos <= 3) )
            top_score_pos = tmodels[i]->nSV[0];
        int top_score_neg = (int) ((1.5 * tmodels[i]->nSV[1])/(nr_class-1) ) ;
        if( (top_score_neg >= ncnt) )
            top_score_neg = tmodels[i]->nSV[1]/ (nr_class-1);
        if( (top_score_neg <= 3) )
            top_score_neg = tmodels[i]->nSV[1];
        svm_node_libsvm *sub_nd = Malloc(svm_node_libsvm ,sub_prob.l);
        double *dec_values = (double*)malloc( (tmodels[i]->nr_class*(tmodels[i]->nr_class-1)/2) * sizeof (double));//memory creating for storing n*(n-1)/2 pairwise decision scores in one-dimensional array
        for(int q=0;q<sub_prob.l;q++)
        {
            sub_nd[q].index = sub_prob.y[q];
            svm_predict_values(tmodels[i],sub_prob.x[q],dec_values);
            sub_nd[q].value = dec_values[0];
        }
        int rvalue = model->MRpos_one_vs_all[weibull_model_count].FitSVM(sub_nd, sub_prob.l, plabel, true, MetaRecognition::positive_model,top_score_pos );//positive tail w.r.t. positive class
        if(rvalue!=1)
            printf("fit weibull positive %d\n",rvalue);
        //model->MRpos_one_vs_all[weibull_model_count].FitSVM(sub_nd, sub_prob.l, plabel, true, MetaRecognition::positive_model,top_score_pos );//positive tail w.r.t. positive class
        if(0) fprintf(stderr,"%lf %lf %lf %lf %lf %lf\n",
                      model->MRpos_one_vs_all[weibull_model_count].W_score(2.0),
                      model->MRpos_one_vs_all[weibull_model_count].W_score(1),
                      model->MRpos_one_vs_all[weibull_model_count].W_score(.1),
                      model->MRpos_one_vs_all[weibull_model_count].W_score(-.1),
                      model->MRpos_one_vs_all[weibull_model_count].W_score(-1),
                      model->MRpos_one_vs_all[weibull_model_count].W_score(-2));
        
        weibull_model_count++;
        free(sub_prob.x);
        free(sub_prob.y);
    }
    if(param->probability){
        info("Probability not yet supported for openset");
        //put prob code hwere when it is
    }
    else
    {
        model->probA=NULL;
        model->probB=NULL;
    }
    /* now we allocate overall SV for the set, and copy data back into the multi-class model */
    int total_sv = 0;
    keep_cnt=0;
    for(i=0;i<nr_class;i++) {
        if(tmodels[i])  {
            total_sv +=  tmodels[i]->l;
            model->rho[keep_cnt] = model->rho[i];
            keep_cnt++;
        } else {
            //      model->label[nr_class-(i-keep_cnt)-1] = model->label[i];// collect negative labels at the end.. there are no models
        }
    }
    info("Total nSV = %d\n",total_sv);
    model->openset_dim = keep_cnt;
    model->l = total_sv;
    int svcnt = 0; /* where are we on overall SV count */
    if(keep_cnt == nr_class){
        model->nSV = Malloc(int ,model->nr_class+1);
        model->SV = Malloc(svm_node *,total_sv);
        model->sv_coef = Malloc(double *,(model->nr_class+1));
        for(i=0;i<nr_class;i++){
            if(tmodels[i]) {
                model->nSV[i] = (int)tmodels[i]->l;
                model->sv_coef[i] = Malloc(double,total_sv);
                memset(model->sv_coef[i],0,total_sv*sizeof(double));
                for(int k=0;k<model->nSV[i];k++,svcnt++){
                    model->SV[svcnt] = tmodels[i]->SV[k];
                    model->sv_coef[i][svcnt] = tmodels[i]->sv_coef[0][k];
                }
            }
        }
        
        for(i=0;i<nr_class;i++){
            if(tmodels[i]) {
                svm_free_model_content(tmodels[i]);
                free(tmodels[i]);
            }
        }
        
    } else {
        /* if we have negative labesl to be ignored, we need to save the models a little differnt */
        if(!(keep_cnt == 1 && nr_class==2)){
            fprintf(stderr,"Openset ignoring  negative classess currently only supported for binary.. did you forget the -B flag?\n"
                    "This may not  build a valid model and will leak memory");
        }
        
        int keepit=0;
        if(tmodels[1]) { 
            keepit=1;
        }
        model->rho[0] = tmodels[keepit]->rho[0];
        model->SV = tmodels[keepit]->SV;
        model->nSV = tmodels[keepit]->nSV;
        model->sv_coef =   tmodels[keepit]->sv_coef;
        
        for(i=0;i<nr_class;i++){
            if(tmodels[i]) {
                //      svm_free_model_content(tmodels[i]);// we reused some of the content, don't free them all,do only those we don't need
                free(tmodels[i]->label);
                free(tmodels[i]->rho);
                free(tmodels[i]);
            }
        }
    }
    
    model->free_sv=0;
    
    free(tmodels);
    
    free(x);
    free(label);
    free(start);
    free(count);
    free(perm);
    
    return model;
}

svm_model *svm_train_onevset_pairs(svm_model* model, const svm_problem *prob, const svm_parameter *param){

  /* mod the tyep so it thinks it's a regular pairwise svm */ 
    model->param.svm_type = C_SVC;
  model->param.do_open=0;
  /* do regular pairwise training */ 

  model = svm_train_binary_pairs(model,prob,  &model->param);

  /* setup model for multiple openset classes, we have n*(n-1)/2 models but n*(n-1) sets of plance  (since its not just a sign flip  */
  model->openset_dim = model->nr_class*model->nr_class;
  model->alpha = Malloc(double,  model->openset_dim);                
  model->omega = Malloc(double,model->openset_dim);              
  memset(model->alpha,0,model->openset_dim*sizeof(double));
  memset(model->omega,0,model->openset_dim*sizeof(double));
  //  model->label = Malloc(int,nr_class); // not needed as train_binary_pair_ already allocate and filled in labels



  if(prob)  openset_analyze_pairs(*prob,  model);

  /* return to openset */ 
  model->param.do_open=1;
  model->param.svm_type=OPENSET_PAIR;

    
  if(param->probability){
    info("Probability not yet supported for openset");
    //put prob code hwere when it is
  }
  else{
      model->probA=NULL;
      model->probB=NULL;
    }
  
  return model;
}

//
// Interface functions
//
svm_model *svm_train(const svm_problem *prob, const svm_parameter *param)
{
	svm_model *model = Malloc(svm_model,1);
        memset(model,0,sizeof(svm_model)); // clear out all fields 
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR){
            svm_train_oneclass_or_regression(model,prob,param);
          }
	else if(param->svm_type == OPENSET_OC){
            svm_train_onevset_oneclass(model,prob,param);
          }
	else if(param->svm_type == OPENSET_BIN){
            svm_train_onevset_binary(model,prob,param);
          }
	else if(param->svm_type == OPENSET_PAIR){
            svm_train_onevset_pairs(model,prob,param);
          }
	else if(param->svm_type == ONE_VS_REST_WSVM){
            svm_train_wsvm_onevset_binary(model,prob,param);
			// mexPrintf("Enters 8\n");
          }
	else if(param->svm_type == ONE_WSVM)
		  {
			// mexPrintf("Enters 9\n");
            svm_train_wsvm_onevset_oneclass(model,prob,param);
          }
    else if(param->svm_type == PI_SVM)
    {
		// mexPrintf("Enters 10\n");
        svm_train_pi_svm_onevset_binary(model,prob,param);
    }
	else  {
            svm_train_binary_pairs(model,prob,param);
          }
	return model;
}

// Stratified cross validation
void svm_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;

	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC || param->svm_type == ONE_WSVM || param->svm_type == PI_SVM) && nr_fold < l){
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++){
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++){
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++){
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++){
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);	
		free(label);
		free(count);	
		free(index);
		free(fold_count);
	}
	else{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++){
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++){
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++){
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++){
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param);
        if(param->svm_type == ONE_WSVM || param->svm_type == PI_SVM)
            submodel->param.openset_min_probability = param->openset_min_probability;
		
        if(param->probability &&
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC)){
			double *prob_estimates=Malloc(double,svm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict_probability(submodel,prob->x[perm[j]],prob_estimates);
			free(prob_estimates);			
		}
		else if(param->svm_type == ONE_CLASS){
                     // undo the hack to make one_class do probabilities
			for(j=begin;j<end;j++)
                          target[perm[j]] = (svm_predict(submodel,prob->x[perm[j]])>0)?1:-1;;
		}
        else if(param->svm_type == ONE_WSVM || param->svm_type == PI_SVM){
            for(j=begin;j<end;j++){
                int *votes = NULL;
                double **scores = Malloc(double *, nr_class+1);
                votes = Malloc(int,nr_class+1);
                for(int v=0; v<nr_class; v++){
                    scores[v] = Malloc(double, nr_class);
                    memset(scores[v],0,nr_class*sizeof(double));
                }
                for(int ii=0;ii<nr_class;ii++){
                    votes[ii]=0;
                    for(int jj=0; jj<nr_class; jj++){
                        scores[ii][jj] = 0;
                    }
                }
                target[perm[j]] = svm_predict_extended(submodel,prob->x[perm[j]], scores, votes);
                //target[perm[j]] = svm_predict_extended(submodel,prob->x[perm[j]],scores,votes);
                //printf("%lf  %lf\n",prob->y[perm[j]],target[perm[j]]);
                //cleanup scores and votes
                for(int v=0; v<nr_class; v++)
                    if(scores[v] != NULL)
                        free(scores[v]);
                if(scores != NULL)
                    free(scores);
                if(votes != NULL)
                    free(votes);
                
            }
            
        }
		else
			for(j=begin;j<end;j++)
				target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}		
	free(fold_start);
	free(perm);	
}

void svm_cross_validation_wsvm(const svm_problem *prob, const svm_parameter *param, const svm_problem *prob_one_wsvm, const svm_parameter *param_one_wsvm ,int nr_fold, double *target)
{
	int i;
	int *fold_start = Malloc(int,nr_fold+1);
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
    //int *label = NULL;
	if(param->svm_type == ONE_VS_REST_WSVM && nr_fold < l){
        int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++){
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++){
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++){
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++){
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);
		free(label);
		free(count);	
		free(index);
		free(fold_count);
	}
	else{
        for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++){
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}
    
    int falsepos=0, falseneg=0, truepos=0, trueneg=0;
    int total;
	for(i=0;i<nr_fold;i++){
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);
        
        struct svm_problem subprob_one_wsvm;
		subprob_one_wsvm.l = l-(end-begin);
		subprob_one_wsvm.x = Malloc(struct svm_node*,subprob_one_wsvm.l);
		subprob_one_wsvm.y = Malloc(double,subprob_one_wsvm.l);
			
		k=0;
		for(j=0;j<begin;j++){
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
            
            subprob_one_wsvm.x[k] = prob_one_wsvm->x[perm[j]];
			subprob_one_wsvm.y[k] = prob_one_wsvm->y[perm[j]];
            
			++k;
		}
		for(j=end;j<l;j++){
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
            
            subprob_one_wsvm.x[k] = prob_one_wsvm->x[perm[j]];
			subprob_one_wsvm.y[k] = prob_one_wsvm->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param);
        struct svm_model *submodel_one_wsvm = svm_train(&subprob_one_wsvm,param_one_wsvm);

        submodel->param.openset_min_probability = param->openset_min_probability;
        submodel_one_wsvm->param.openset_min_probability = param_one_wsvm->openset_min_probability;
        
		for(j=begin;j<end;j++){
			int *votes = NULL;
			double **scores = Malloc(double *, nr_class+1);
                        votes = Malloc(int,nr_class+1);
			for(int v=0; v<nr_class; v++){
				scores[v] = Malloc(double, nr_class);
                              	memset(scores[v],0,nr_class*sizeof(double));
			}
			for(int ii=0;ii<nr_class;ii++){
				votes[ii]=0;
				for(int jj=0; jj<nr_class; jj++){
					scores[ii][jj] = 0;					
				}
			}
            target[perm[j]] = svm_predict_extended_plus_one_wsvm(submodel,submodel_one_wsvm,prob->x[perm[j]], scores, votes);
		}
        svm_free_and_destroy_model(&submodel_one_wsvm);
		svm_free_and_destroy_model(&submodel);
        free(subprob_one_wsvm.x);
		free(subprob_one_wsvm.y);
		free(subprob.x);
		free(subprob.y);
	}
    
    //free(label);
	free(fold_start);
	free(perm);	
}

int svm_get_svm_type(const svm_model *model){
	return model->param.svm_type;
}

int svm_get_nr_class(const svm_model *model){
	return model->nr_class;
}

void svm_get_labels(const svm_model *model, int* label){
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

double svm_get_svr_probability(const svm_model *model){
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
            return model->probA[0];
	else{
            fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
            return 0;
	}
}

//Wraps old svm_predict_values func so calls to old method signature don't break 
double svm_predict_values(const svm_model *model, const svm_node *x, double* dec_values)
{
	int *vote = NULL;
	double **scores = Malloc(double *, model->nr_class);
	for(int v=0; v<model->nr_class; v++){
		scores[v] = Malloc(double, model->nr_class);
		for(int z=0; z<model->nr_class; z++)
			scores[v][z] = 0;
	}

	double pred_result = svm_predict_values_extended(model,x,dec_values, scores, vote);

	if(vote != NULL)
		free(vote);

	for(int v=0; v<model->nr_class; v++)
		if(scores[v] != NULL)
			free(scores[v]);

	if(scores != NULL)
		free(scores);

	return pred_result;
}
//adds new parameters to old func, gets wrapped to preserve backwards compatability 
double svm_predict_values_extended(const svm_model *model, const svm_node *x, 
								   double*& dec_values, double **&scores, int*& vote)
{
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR){
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		for(int i=0;i<model->l;i++)
			sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
		sum -= model->rho[0];
		if(dec_values) *dec_values = sum; // decision scores stored here.

		if(model->param.svm_type == ONE_CLASS)
			//return (sum>0)?1:-1;
			return sum; //have to return score rather than +1 or -1 in one-class svm
		else
			return sum;
	}
	else if(model->param.svm_type == OPENSET_OC ||
                model->param.svm_type == OPENSET_BIN ){
        double mindist=-999999;
        int bestindex = model->param.rejectedID;
        int treat_as_binary=0;
        int negindex=bestindex;
        if(model->nr_class==2 && model->openset_dim ==1){
            treat_as_binary = 1;
            for(int j=0;j<model->nr_class;j++){
                if(model->label[j]<0) negindex=j;
            }
        }

        for(int j=0;j<model->openset_dim;j++){
            double *sv_coef = model->sv_coef[j];
            double omega = model->omega[j];
            double alpha = model->alpha[j];
            double rho = model->rho[j];
            double sum = 0;
            for(int i=0;i<model->l;i++){
                if(sv_coef[i] != 0)
                    sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
            }
            double dist = (sum - rho);
            double t1 = omega-dist;
            double t2 = dist-alpha;
            if(t2 <0 &&  bestindex== model->param.rejectedID)
                bestindex = negindex;
            // we only enfource second plane for linear, so we work around for others but using only one distance
            if(model->param.kernel_type != LINEAR) t1 =t2;
            //convert it so we return positive if in slab, and negative if outside (and distance is to nearest plane)
            if (t2 <= 0 )  dist = t2;
            else if (t1 <= 0)  dist = t1;
            else if(t1 > t2) dist = t2;
            else dist =t1;
            if(omega-alpha >0) {
                dist = dist/(omega-alpha);  // we normalize by slab width..
                if(scores) scores[j][0]=dist;
                if(dec_values) dec_values[j] = dist;
                if(t1>0 && vote) vote[j]=1;
                if(mindist < dist && dist >0) {
                    mindist = dist;
                    bestindex = j;
                }
            }
            else if(treat_as_binary>0){
                if(mindist <0)
                    bestindex =j;
            }
        }
        if(bestindex == model->param.rejectedID)
            return model->param.rejectedID;
        else
            return model->label[bestindex];
    } 
    else if(model->param.svm_type == ONE_WSVM ){
        for(int j=0;j<model->openset_dim;j++){
            //for one wsvm
            double *sv_coef = model->sv_coef[j];
            double rho = model->rho[j];
            double sum = 0;	
            for(int i=0;i<model->l;i++){
            //for wsvm
                if(sv_coef[i] != 0)
                sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
            }		   	
            double dist = sum - rho;
            dec_values[j] = dist;
            double pos_class_pos_score = model->MRpos_one_class[j].W_score(dec_values[j]);//positive tail using on class wsvm
            double tscore = pos_class_pos_score;		    
            if(scores){
              scores[j][0] =  tscore;	
            }	  
	   }
   	   double max_prob=scores[0][0];int max_prob_index=0;
	   for(int jj=0; jj< model->openset_dim; jj++){
		if(scores[jj][0] > max_prob){
			max_prob = scores[jj][0];
			max_prob_index = jj;
		}
	   }
	  return model->label[max_prob_index];
    }
    else if(model->param.svm_type == PI_SVM ){
        for(int j=0;j<model->openset_dim;j++){
            double *sv_coef = model->sv_coef[j];
            double rho = model->rho[j];
            double sum = 0;
            for(int i=0;i<model->l;i++){
                if(sv_coef[i] != 0)
                    sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
            }
            double dist = sum - rho;
            dec_values[j] = dist;
            double pos_class_pos_score = model->MRpos_one_vs_all[j].W_score(dec_values[j]);//positive tail w.r.t. positive class
            double tscore = pos_class_pos_score;
            if(scores){
                scores[j][0] =  tscore;
            }
        }
        double max_prob=scores[0][0];int max_prob_index=0;
        for(int jj=0; jj< model->openset_dim; jj++){
            if(scores[jj][0] > max_prob){
                max_prob = scores[jj][0];
                max_prob_index = jj;
            }
        }
        return model->label[max_prob_index];
    }
	else  if(model->param.svm_type == OPENSET_PAIR ){
        int i;
        int nr_class = model->nr_class;
        int l = model->l;
          
        double *kvalue = Malloc(double,l);
        for(i=0;i<l;i++)
            kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

        int *start = Malloc(int,nr_class);
        start[0] = 0;
        for(i=1;i<nr_class;i++)
        start[i] = start[i-1]+model->nSV[i-1];
          
        //		int *vote = Malloc(int,nr_class);
        vote = Malloc(int,nr_class);
        for(i=0;i<nr_class;i++)
            vote[i] = 0;
          
        for(i=0;i<nr_class*(nr_class-1)/2;i++)
            dec_values[i] = model->param.rejectedID;
        int p=0;
        double maxdist=-1;
        int bestindex=model->param.rejectedID;
        for(i=0;i<nr_class;i++){
            for(int j=i+1;j<nr_class;j++){
                double sum = 0;
                int si = start[i];
                int sj = start[j];
                int ci = model->nSV[i];
                int cj = model->nSV[j];
                  
                int k;
                double *coef1 = model->sv_coef[j-1];
                double *coef2 = model->sv_coef[i];
                for(k=0;k<ci;k++)
                sum += coef1[si+k] * kvalue[si+k];
                for(k=0;k<cj;k++)
                    sum += coef2[sj+k] * kvalue[sj+k];
                sum -= model->rho[p];
                /* we do ij and ji separately (since they can different planes driven by  different "sets"  */
                double dist = sum;
                double omega = model->omega[i*nr_class+j];
                double alpha = model->alpha[i*nr_class+j];
                double t1 = omega-dist;
                double t2 = dist-alpha;
                  /* we only enfource second plane for linear, so we work around for others but using only one distance  */ 
                if(model->param.kernel_type != LINEAR) t1 =t2;
                  /*convert it so we return positive if in slab, and negative if outside (and distance is to nearest plane)*/
                if (t2 <= 0 )  dist = t2;
                else if (t1 <= 0)  dist = t1;
                else if(t1 > t2) dist = t2;
                else dist =t1;
                if(model->param.kernel_type == LINEAR)
                dist = dist/(omega-alpha);  // we normalize by slab width..
                scores[i][j] = dist;
                if(dist >maxdist){
                    maxdist = dist;
                    bestindex=i;
                }
                dist = sum;
                omega = model->omega[j*nr_class+i];
                alpha = model->alpha[j*nr_class+i];
                t1 = omega-dist;
                t2 = dist-alpha;
                /* we only enfource second plane for linear, so we work around for others but using only one distance */
                if(model->param.kernel_type != LINEAR) t1 =t2;
                /*convert it so we return positive if in slab, and negative if outside (and distance is to nearest plane)*/
                if (t2 <= 0 )  dist = t2;
                else if (t1 <= 0)  dist = t1;
                else if(t1 > t2) dist = t2;
                else dist =t1;
                if(model->param.kernel_type == LINEAR)
                dist = dist/(omega-alpha);  // we normalize by slab width..
                scores[j][i] = dist;

                if(dist >maxdist){
                maxdist = dist;
                bestindex=j;
                }
                /* only the winner gets a vote, and even then only if the scores are positive */
                if(scores[i][j] > dist) {
                    dist = scores[i][j];
                    if(scores[i][j] >0) ++vote[i];
                } else  if(scores[j][i] >0) ++vote[j];

                dec_values[p] = dist;
                p++;
            }
        }
        free(kvalue);
        free(start);

        if(bestindex != model->param.rejectedID) vote[bestindex] += 10;

        int vote_max_idx = 0;
        int maxvote=-1;
        int maxlab=model->param.rejectedID;;
        for(i=0;i<nr_class;i++){
            if(vote[i] > maxvote){
                vote_max_idx = i;
                maxvote = vote[i];
            }
        }
        if(maxvote>0)
        maxlab = model->label[vote_max_idx];
        return maxlab;
        
	}else{ /* classic binary pair voting */
		int i;
		int nr_class = model->nr_class;
		int l = model->l;
		
		double *kvalue = Malloc(double,l);
		for(i=0;i<l;i++)
			kvalue[i] = Kernel::k_function(x,model->SV[i],model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

                if(vote==NULL) vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			dec_values[i] = model->param.rejectedID;
		int p=0;
		for(i=0;i<nr_class;i++){
			for(int j=i+1;j<nr_class;j++){
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];
				
				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				//save in both bins, examples 1-2 and 2-1
				scores[i][j] = dec_values[p];
				scores[j][i] = -dec_values[p];
				p++;
			}
		}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);

                  return model->label[vote_max_idx];
	}
        return -99999.;
}





//adds new parameters to old func, gets wrapped to preserve backwards compatability 
double svm_predict_values_extended_plus_one_wsvm(const svm_model *model,const svm_model *model_one_wsvm, const svm_node *x, 
								   double*& dec_values_wsvm,double*& dec_values_one_wsvm, double **&scores, int*& vote){
	// mexPrintf("in svm_predict_values_extended_plus_one_wsvm: SVM type %d %d\n", model->param.svm_type, model_one_wsvm->param.svm_type);
    int nr_class = model->nr_class;
    if(model->param.svm_type == ONE_VS_REST_WSVM )
	{
	  double *one_class_wscores = Malloc(double, nr_class+1);
	  for(int i=0;i<nr_class;i++)
          one_class_wscores[i] = 0;
      double sum_one_class = 0;
      for(int j=0;j<model->openset_dim;j++)
	  {
	    // for one class wsvm	
        double *sv_coef_one_wsvm = model_one_wsvm->sv_coef[j];
        double rho_one_wsvm = model_one_wsvm->rho[j];
		// mexPrintf("rho_one_wsvm: %f\n", rho_one_wsvm);
        double sum_one_wsvm = 0;

	    //for wsvm
        double *sv_coef = model->sv_coef[j];
        double rho = model->rho[j];
		// mexPrintf("rho: %f\n", rho);
        double sum = 0;
        for(int i=0;i<model_one_wsvm->l;i++)
		{			
			//for one class wsvm	
			if(sv_coef_one_wsvm[i] != 0)
			{
				sum_one_wsvm += sv_coef_one_wsvm[i] * Kernel::k_function(x,model_one_wsvm->SV[i],model_one_wsvm->param);
				// mexPrintf("sv_coef_one_wsvm[i] %d: %f\n", i, sv_coef_one_wsvm[i]);
			}
        }
        for(int i=0;i<model->l;i++)
		{
			//for wsvm	
			if(sv_coef[i] != 0)
			{
                sum += sv_coef[i] * Kernel::k_function(x,model->SV[i],model->param);
				// mexPrintf("sv_coef[i] %d: %f\n", i, sv_coef[i]);
			}
        }
	    double dist_one_wsvm = sum_one_wsvm - rho_one_wsvm;		   	
        double dist = sum - rho;
		// mexPrintf("dist_one_wsvm it %d: %f\n", j, dist_one_wsvm);
		// mexPrintf("dist it %d: %f\n", j, dist);
		
	    dec_values_wsvm[j] = dist;
		//mexPrintf("dec_values_wsvm[j]: %f\n", dec_values_wsvm[j]);
		// mexPrintf("sign: %d\n", model->MRpos_one_vs_all[j].sign);
		// mexPrintf("translate_amount: %f\n", model->MRpos_one_vs_all[j].translate_amount);
		// mexPrintf("small_score: %f\n", model->MRpos_one_vs_all[j].small_score);
		// mexPrintf("parmhat[0]: %f\n", model->MRpos_one_vs_all[j].parmhat[0]);
		// mexPrintf("parmhat[1]: %f\n", model->MRpos_one_vs_all[j].parmhat[1]);
		// mexPrintf("ftype: %d\n", (int)model->MRpos_one_vs_all[j].ftype);
	    dec_values_one_wsvm[j] = dist_one_wsvm;	
		// mexPrintf("dec_values_one_wsvm[j]: %f\n", dec_values_one_wsvm[j]);
	    double pos_class_pos_score = model->MRpos_one_vs_all[j].W_score(dec_values_wsvm[j]); // positive tail w.r.t. positive class
		//mexPrintf("pos_class_pos_score: %f\n", pos_class_pos_score);
	    double pos_class_comp_score = model->MRcomp_one_vs_all[j].W_score(dec_values_wsvm[j]); // compliment tail w.r.t. positive class
		//mexPrintf("pos_class_comp_score: %f\n", pos_class_comp_score);
	    double pos_one_class_pos_score = one_class_wscores[j] = model_one_wsvm->MRpos_one_class[j].W_score(dec_values_one_wsvm[j]);//positive tail using on class wsvm  
		// mexPrintf("pos_one_class_pos_score: %f\n", pos_one_class_pos_score);
		
		/*
		double translated_x = dec_values_wsvm[j]* model_one_wsvm->MRpos_one_class[j].sign +  model_one_wsvm->MRpos_one_class[j].translate_amount - model_one_wsvm->MRpos_one_class[j].small_score;
		mexPrintf("translated_x: %f\n", translated_x);
		double tempVal, tempVal1;
		if(translated_x < 0) 
			mexPrintf("translated_x < 0! error!\n");
		tempVal =  translated_x/model_one_wsvm->MRpos_one_class[j].parmhat[0];
		mexPrintf("tempVal: %f\n", tempVal);
		tempVal1 = pow(tempVal, model_one_wsvm->MRpos_one_class[j].parmhat[1]);
		mexPrintf("tempVal1: %f\n", tempVal1);
		double wscore = 1-exp(-1*tempVal1);
		mexPrintf("wscore: %f\n", wscore);
		if((int)model_one_wsvm->MRpos_one_class[j].ftype==3 || (int)model_one_wsvm->MRpos_one_class[j].ftype==4) 
			wscore = 1 - wscore;
		mexPrintf("wscore after -1: %f\n", wscore);
		*/
		
	    sum_one_class+= pos_one_class_pos_score; 	
		// mexPrintf("sum_one_class: %f\n", sum_one_class);

        double tscore = pos_class_pos_score*pos_class_comp_score;
	    if(scores)
		{
			scores[j][0] =  tscore;	
			mexPrintf("tscore %d: %f\n", j, tscore);
		}
	   }
       bool one_class_flag = false;
	   for(int j=0;j<model->openset_dim;j++)
	   {
		   // mexPrintf("one_class_wscores[j]: %f\n", one_class_wscores[j]);
           if(one_class_wscores[j] <= model_one_wsvm->param.openset_min_probability)
               one_class_flag = true ;
           else
               one_class_flag = false;
			one_class_flag = false; // No rejection!
           if(one_class_flag)
		   {
			   // mexPrintf("RESET SCORE!\n");
               scores[j][0] = 0;
		   }
           else
               scores[j][0] = scores[j][0];
	   }
   	   double max_prob=scores[0][0];int max_prob_index=0;
	   for(int jj=0; jj< model->openset_dim; jj++)
	   {
			if(scores[jj][0] > max_prob)
			{
				max_prob = scores[jj][0];
				max_prob_index = jj;
			}
	   }
	   // mexPrintf("max_prob_index: %f\n", max_prob_index);
	   free(one_class_wscores);
       return model->label[max_prob_index];
    }
    else
	{
            fprintf(stderr,"Given WSVM does not have relavant model files for One-Class WSVM + WSVM" );
            exit(1);
	}
}

double svm_predict(const svm_model *model, const svm_node *x)
{
	int *vote = NULL;
	double **scores = NULL;
	scores = Malloc(double *, model->nr_class);
	for(int v=0; v<model->nr_class; v++){
		scores[v] = Malloc(double, model->nr_class);
		for(int z=0; z<model->nr_class; z++)
			scores[v][z] = 0;
	}
	double pred_result = svm_predict_extended(model, x , scores, vote);
	for(int v=0; v<model->nr_class; v++)
          if(scores[v] != NULL)
            free(scores[v]);
	if(scores != NULL)
		free(scores);
	if(vote != NULL)
		free(vote);
	return pred_result;
}

double svm_predict_extended(const svm_model *model, const svm_node *x,
						    double **&scores, int *&vote)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
          dec_values = Malloc(double, 1);
	else if(model->param.svm_type == OPENSET_BIN ||
            model->param.svm_type == ONE_WSVM ||
            model->param.svm_type == ONE_VS_REST_WSVM ||
            model->param.svm_type == PI_SVM)
        dec_values = Malloc(double, nr_class+1);
    else
          dec_values = Malloc(double, (nr_class*(nr_class-1)/2));
	double pred_result = svm_predict_values_extended(model, x, dec_values, scores, vote);
	if ( model->param.svm_type == ONE_CLASS ){
		if(pred_result > 0)	pred_result = 1;
		else			pred_result = -1;	
	}
	free(dec_values);
	return pred_result;
}
double svm_predict_extended_plus_one_wsvm(const svm_model *model,const svm_model *model_one_wsvm, const svm_node *x,
						    double **&scores, int *&vote){
	int nr_class = model->nr_class;
	double *dec_values_wsvm,*dec_values_one_wsvm;
	dec_values_one_wsvm = Malloc(double, nr_class+1);
	if(model->param.svm_type == ONE_VS_REST_WSVM )
		dec_values_wsvm = Malloc(double, nr_class+1);
	else
		dec_values_wsvm = Malloc(double, (nr_class*(nr_class-1)/2));

	// mexPrintf("in svm_predict_extended_plus_one_wsvm\n");;
	double pred_result = svm_predict_values_extended_plus_one_wsvm(model,model_one_wsvm,x, dec_values_wsvm,dec_values_one_wsvm, scores, vote);
	free(dec_values_wsvm);
	free(dec_values_one_wsvm);
	return pred_result;
}

double svm_predict_probability(
	const svm_model *model, const svm_node *x, double *prob_estimates){
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL){
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, (nr_class*(nr_class-1)/2));
		svm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++){
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);	     
		return model->label[prob_max_idx];
	}
	else 
		return svm_predict(model, x);
}

static const char *svm_type_table[] ={
  "c_svc","nu_svc","one_class","epsilon_svr","nu_svr","openset_oc", "openset_pair", "openset_bin", "one_vs_rest_wsvm", "one_wsvm","pi_svm",NULL
};

static const char *kernel_type_table[]={
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int svm_save_model(const char *model_file_name, const svm_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	const svm_parameter& param = model->param;

        fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %22.20g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", param.coef0);


	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "openset_dim %d\n", model->openset_dim);
	fprintf(fp, "total_sv %d\n",l);
	
	if(model->label){
          fprintf(fp, "label");
          for(int i=0;i<nr_class;i++)
            fprintf(fp," %d",model->label[i]);
          fprintf(fp, "\n");
	}
	if(model->rho){
          fprintf(fp, "rho");
          int rhosize=(nr_class*(nr_class-1)/2);
          if(model->openset_dim >0 && model->openset_dim < rhosize ) rhosize = model->openset_dim;
          for(int i=0; i< rhosize; i++)  fprintf(fp," %24.20g",model->rho[i]);
          fprintf(fp, "\n");
	}

	if (param.svm_type == OPENSET_OC 
		|| param.svm_type == ONE_WSVM	
		 || param.svm_type == OPENSET_BIN 
		 || param.svm_type == OPENSET_PAIR
		 || param.svm_type == ONE_VS_REST_WSVM
		 || param.svm_type == PI_SVM)
	{
	  if(param.neg_labels)
		fprintf(fp, "Neg_labels 1\n");
	  else 
		fprintf(fp, "Neg_labels 0\n");
	  fprintf(fp, "Rejected_ID %d\n", param.rejectedID);
	  if(model->alpha != NULL) {
		fprintf(fp,"alpha ");
		for(int i=0; i< model->openset_dim; i++) fprintf(fp," %24.20g", model->alpha[i]);
		fprintf(fp,"\n");
	  }
	  if(model->omega != NULL) {
		fprintf(fp,"omega ");
		for(int i=0; i< model->openset_dim; i++) fprintf(fp," %24.20g", model->omega[i]);
		fprintf(fp,"\n");
	  }
	}
	//for ONE_VS_REST_WSVM
	if (  (param.svm_type == ONE_VS_REST_WSVM) && ((model->MRpos_one_vs_all != NULL) && (model->MRcomp_one_vs_all != NULL)) )
	{
	  fprintf(fp,"MR_pos_one_vs_all ");
	  for(int i=0; i< model->openset_dim; i++) model->MRpos_one_vs_all[i].Save(fp);

	  fprintf(fp,"MR_comp_one_vs_all ");
	  for(int i=0; i< model->openset_dim; i++) model->MRcomp_one_vs_all[i].Save(fp);

	  //fprintf(fp,"MR_pos_one_class ");
	  //for(int i=0; i< model->openset_dim; i++) model->MRpos_one_class[i].Save(fp);

	}
	//for ONE_WSVM
	if ( (param.svm_type == ONE_WSVM) && (model->MRpos_one_class != NULL)  )
	{
	  fprintf(fp,"MR_pos_one_class ");
	  for(int i=0; i< model->openset_dim; i++) model->MRpos_one_class[i].Save(fp);

	}
	//for PI_SVM
	if ((param.svm_type == PI_SVM && ( (model->MRpos_one_vs_all != NULL) ) ))
	{
		fprintf(fp,"MR_pos_one_vs_all ");
		for(int i=0; i< model->openset_dim; i++) model->MRpos_one_vs_all[i].Save(fp);
	}
	
    // regression has probA only
	if(model->probA){
		fprintf(fp, "probA");
		for(int i=0;i<(nr_class*(nr_class-1)/2);i++)
			fprintf(fp," %g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB){
		fprintf(fp, "probB");
		for(int i=0;i<(nr_class*(nr_class-1)/2);i++)
			fprintf(fp," %g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV){
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const svm_node * const *SV = model->SV;

        int limit = nr_class-1;
        if(model->param.svm_type == OPENSET_OC || model->param.svm_type == ONE_WSVM|| model->param.svm_type == OPENSET_BIN || model->param.svm_type == ONE_VS_REST_WSVM || model->param.svm_type == PI_SVM)
          limit = model->openset_dim;
	for(int i=0;i<l;i++){
		for(int j=0;j<limit;j++)
			fprintf(fp, "%.16g ",sv_coef[j][i]);

	const svm_node *p = SV[i];

	if(param.kernel_type == PRECOMPUTED)
		fprintf(fp,"0:%d ",(int)(p->value));
	else
		while(p->index != -1)
		{
			fprintf(fp,"%d:%.12g ",p->index,p->value);
			p++;
		}
	fprintf(fp, "\n");
	}
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}


void svm_badload_cleanup(svm_model *model){
  if(model){
    if(	model->rho != NULL)  free(model->rho);
    if(	model->probA != NULL)  free(model->probA);
    if(	model->probB != NULL)  free(model->probB);
    if(	model->label != NULL)  free(model->label);
    if(	model->nSV != NULL)  free(model->nSV);
    if(	model->alpha != NULL)  free(model->alpha);
    if(	model->omega != NULL)  free(model->omega);

    if(	model->MRpos_one_vs_all != NULL)  free(model->MRpos_one_vs_all);
    if(	model->MRcomp_one_vs_all != NULL)  free(model->MRcomp_one_vs_all);  
    if(	model->MRpos_one_class != NULL)  free(model->MRpos_one_class);
    free(model);
  }
}


svm_model *svm_load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"rb");
	if(fp==NULL) return NULL;
	
	// read parameters

	svm_model *model = Malloc(svm_model,1);
	svm_parameter& param = model->param;
	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;
	
    // EXTRA OPEN_SET
	model->alpha = NULL;
    model->omega = NULL;
	model->MRpos_one_vs_all = NULL;
	model->MRcomp_one_vs_all = NULL;  
	model->MRpos_one_class = NULL;

    model->openset_dim = 0;
    model->param.rejectedID = -9999; /* id for rejected classes (-99999 is the default) */
    model->param.neg_labels=false;
	// EXTRA OPEN_SET (END)
	
	while(1)
	{
	char cmd[82]={0};
		fscanf(fp,"%80s",cmd);

		if(strcmp(cmd,"svm_type")==0){
                  fscanf(fp,"%80s",cmd);
                  int i;
                  for(i=0;svm_type_table[i];i++){
                      if(strcmp(svm_type_table[i],cmd)==0){
                          param.svm_type=i;
                          break;
                        }
                    }   	
                if(svm_type_table[i] == NULL){
                      fprintf(stderr,"unknown svm type.\n");
                      svm_badload_cleanup(model);
                      return NULL;
                    }
                }
		else if(strcmp(cmd,"kernel_type")==0){		
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;kernel_type_table[i];i++){
				if(strcmp(kernel_type_table[i],cmd)==0){
					param.kernel_type=i;
					break;
				}
			}
			if(kernel_type_table[i] == NULL){
				fprintf(stderr,"unknown kernel function.\n");
                                svm_badload_cleanup(model);
				return NULL;
			}
		}
		else if(strcmp(cmd,"degree")==0)
			fscanf(fp,"%d",&param.degree);
		else if(strcmp(cmd,"gamma")==0)
			fscanf(fp,"%lf",&param.gamma);
		else if(strcmp(cmd,"coef0")==0)
			fscanf(fp,"%lf",&param.coef0);
		else if(strcmp(cmd,"nr_class")==0)
                  fscanf(fp,"%d",&model->nr_class);
		else if(strcmp(cmd,"openset_dim")==0)
                  fscanf(fp,"%d",&model->openset_dim);
		else if(strcmp(cmd,"total_sv")==0)
			fscanf(fp,"%d",&model->l);
		else if(strcmp(cmd,"label")==0){
			int n = model->nr_class;
			model->label = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->label[i]);
		}
		else if(strcmp(cmd,"rho")==0){
                  int n=(model->nr_class*(model->nr_class-1)/2);
                  if(model->openset_dim >0 &&  model->openset_dim < n) n = model->openset_dim;
                  model->rho = Malloc(double,n);
                  for(int i=0;i<n;i++)
                    fscanf(fp,"%lf",&model->rho[i]);
		}

		else if(strcmp(cmd,"Neg_labels")==0){
                  int junk;
                  fscanf(fp,"%d",&junk);
                  param.neg_labels=false;
                  if(junk>0) param.neg_labels=true;
                }
		else if(strcmp(cmd,"Rejected_ID")==0){
                  fscanf(fp,"%d",&param.rejectedID);
                }
		else if(strcmp(cmd,"alpha")==0){
                  int n=model->openset_dim;
                  if(n>0){
                    model->alpha = Malloc(double,n);
                    for(int i=0;i<n;i++) 
		      fscanf(fp,"%lf",&model->alpha[i]);
                  }
                  else {
                    fprintf(stderr,"Openset alpha found for non openset class, ignoring it.\n");
                    char junk[10*4096];
                    fgets(junk,10*4096,fp); /* eat the line */
                  }
		}
		else if(strcmp(cmd,"omega")==0){
                  int n=model->openset_dim;
                  if(n>0){
                    model->omega = Malloc(double,n);
                    for(int i=0;i<n;i++) 
		      fscanf(fp,"%lf",&model->omega[i]);
                  }
                  else {
                    fprintf(stderr,"Openset omega found for non openset class, ignoring it.\n");
                    char junk[10*4096];
                    fgets(junk,10*4096,fp); /* eat the line */
                  }
		}
		//for ONE_VS_REST_WSVM POSITIVE
		else if(strcmp(cmd,"MR_pos_one_vs_all")==0){
            int n=model->openset_dim;
            if(n>0 && ( (param.svm_type == ONE_VS_REST_WSVM) || (param.svm_type == PI_SVM) ) ){
                model->MRpos_one_vs_all = new MetaRecognition[model->openset_dim];
                for(int i=0;i<n;i++) {
                    model->MRpos_one_vs_all[i].Load(fp);
                }
            }
            else {
                fprintf(stderr,"Openset MR_pos_one_vs_all found for non ONE_VS_REST WSVM class, ignoring it.\n");
                char junk[10*4096];
                fgets(junk,10*4096,fp); // eat the line
            }
		}
		//for ONE_VS_REST_WSVM COMPLIMENT
		else if(strcmp(cmd,"MR_comp_one_vs_all")==0){
            int n=model->openset_dim;
            if(n>0 &&  (param.svm_type == ONE_VS_REST_WSVM)  ){
                model->MRcomp_one_vs_all = new MetaRecognition[model->openset_dim];
                for(int i=0;i<n;i++) {
                    model->MRcomp_one_vs_all[i].Load(fp);
                }
            }
            else {
                fprintf(stderr,"Openset MR_comp_one_vs_all found for non ONE_VS_REST WSVM class, ignoring it.\n");
                char junk[10*4096];
                fgets(junk,10*4096,fp); // eat the line
            }
		}
		//for ONE_WSVM POSITIVE
		else if(strcmp(cmd,"MR_pos_one_class")==0){
            int n=model->openset_dim;
            if(n>0 && param.svm_type == ONE_WSVM){
                model->MRpos_one_class = new MetaRecognition[model->openset_dim];
                for(int i=0;i<n;i++) {
                    model->MRpos_one_class[i].Load(fp);
                }
            }
            else {
                fprintf(stderr,"Openset MRpos_one_class found for non ONE_WSVM class, ignoring it.\n");
                char junk[10*4096];
                fgets(junk,10*4096,fp); // eat the line
            }
		}
		else if(strcmp(cmd,"probA")==0){
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probA = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probA[i]);
		}
		else if(strcmp(cmd,"probB")==0){
			int n = model->nr_class * (model->nr_class-1)/2;
			model->probB = Malloc(double,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%lf",&model->probB[i]);
		}
		else if(strcmp(cmd,"nr_sv")==0){
			int n = model->nr_class;
			model->nSV = Malloc(int,n);
			for(int i=0;i<n;i++)
				fscanf(fp,"%d",&model->nSV[i]);
		}
		else if(strcmp(cmd,"SV")==0){
			while(1){
				int c = getc(fp);
				if(c==EOF || c=='\n') break;	
			}
			break;
		}
		else{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
                        svm_badload_cleanup(model);
			return NULL;
		}
	}

        /* check stuff we need is there */ 


	// read sv_coef and SV

	int elements = 0;
	long pos = ftell(fp);

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	char *p,*endptr,*idx,*val;

	while(readline(fp)!=NULL){
		p = strtok(line,":");
		while(1)
		{
			p = strtok(NULL,":");
			if(p == NULL)
				break;
			++elements;
		}
	}
	elements += model->l;

	fseek(fp,pos,SEEK_SET);

	int m = model->nr_class - 1;
        if(param.svm_type == OPENSET_OC ||param.svm_type == ONE_WSVM || param.svm_type == OPENSET_BIN || param.svm_type == ONE_VS_REST_WSVM || param.svm_type == PI_SVM) m++;
	int l = model->l;
	model->sv_coef = Malloc(double *,m);
	int i;
	for(i=0;i<m;i++)
          model->sv_coef[i] = Malloc(double,l);
	model->SV = Malloc(svm_node*,l);
	svm_node *x_space = NULL;
	if(l>0) x_space = Malloc(svm_node,elements);

	int j=0;
	for(i=0;i<l;i++){
		readline(fp);
		model->SV[i] = &x_space[j];

		p = strtok(line, " \t");
		model->sv_coef[0][i] = strtod(p,&endptr);
		for(int k=1;k<m;k++){
			p = strtok(NULL, " \t");
			model->sv_coef[k][i] = strtod(p,&endptr);
		}

		while(1){
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if(val == NULL)
				break;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			x_space[j].value = strtod(val,&endptr);

			++j;
		}
		x_space[j++].index = -1;
	}
	free(line);

	if (ferror(fp) != 0 || fclose(fp) != 0)
		return NULL;

	model->free_sv = 1;	// XXX
	return model;
}

void svm_free_model_content(svm_model* model_ptr)
{	
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
		free((void *)(model_ptr->SV[0]));
	if(model_ptr->sv_coef){
          int limit=model_ptr->nr_class-1;
          if(model_ptr->param.svm_type == OPENSET_OC ||model_ptr->param.svm_type == ONE_WSVM || model_ptr->param.svm_type == OPENSET_BIN || model_ptr->param.svm_type == ONE_VS_REST_WSVM || model_ptr->param.svm_type == PI_SVM)
            limit = model_ptr->openset_dim;
	  if(model_ptr->param.svm_type == C_SVC)
		limit = model_ptr->nr_class-1;	
          for(int i=0;i<limit;i++)
            free(model_ptr->sv_coef[i]);
	}
	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label= NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB= NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void svm_free_and_destroy_model(svm_model** model_ptr_ptr){
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL){
		svm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void svm_destroy_param(svm_parameter* param){
	free(param->weight_label);
	free(param->weight);
}

const char *svm_check_parameter(const svm_problem *prob, const svm_parameter *param){
	// svm_type
	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != OPENSET_OC &&
	   svm_type != OPENSET_PAIR &&
	   svm_type != OPENSET_BIN &&
	   svm_type != ONE_VS_REST_WSVM &&
	   svm_type != ONE_WSVM &&
       svm_type != PI_SVM &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";        
	
	// kernel_type, degree
	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->gamma < 0)
		return "gamma < 0";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == OPENSET_BIN ||
	   svm_type == ONE_VS_REST_WSVM ||
       svm_type == PI_SVM ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == OPENSET_OC ||
	   svm_type == ONE_WSVM ||	
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";

	if(param->probability == 1 &&
	   svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";


	// check whether nu-svc is feasible
	
	if(svm_type == NU_SVC){
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++){
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j]){
					++count[j];
					break;
				}
			if(j == nr_class){
				if(nr_class == max_nr_class){
					max_nr_class *= 2;
					label = (int *)realloc(label,(ulong)max_nr_class*sizeof(int));
					count = (int *)realloc(count,(ulong)max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}
	
		for(i=0;i<nr_class;i++){
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++){
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2)){
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;}
           

int svm_check_probability_model(const svm_model *model){
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA!=NULL && model->probB!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void svm_set_print_string_function(void (*print_func)(const char *)){
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}



double openset_compute_risk(double far, double near, double pos_width, double fmeasure){
  if(fmeasure == 0 ) return 1e99;
  if(fmeasure < 0 ) {
    fprintf(stderr,"\n\n openset ?HUH? negative fmeasures %g, not allowed risk =1e99\n Somthing is probably very wrong \n\n", fmeasure);
    return 1e99;
  }

  /* nearly empty plane has very high risk in other way */ 
  double width = far - near;
  double mwidth =.0001;
  if(width < mwidth) width = mwidth;
  if(pos_width < mwidth) pos_width = mwidth;
  double gen_ratio = width/pos_width;
  double spec_ratio = pos_width/width;
  if(gen_ratio < 1) gen_ratio=1;
  if(spec_ratio < 1) spec_ratio=1;
  double risk = (gen_ratio +  spec_ratio )  + 2/fmeasure;
  return  risk;

}

double openset_error_for_optimization(double far, double near , double pos_width, int true_pos, int true_neg, int false_pos, int false_neg, struct svm_parameter &param, 	int optimize= OPT_BALANCEDRISK){
  double precision=0, recall=0;
  optimize = param.optimize;
  if ((true_pos+false_pos) > 0)
    precision = ((double) (true_pos)/(true_pos+false_pos));
  if((true_pos + false_neg) > 0)
    recall = ((double) true_pos)/(true_pos + false_neg);

  double fmeasure = 0;
  if(param.beta*param.beta*precision + recall>0)  
    fmeasure = (1+param.beta*param.beta) *precision*recall/(param.beta*param.beta*precision + recall);
  double risk = openset_compute_risk(far, near, pos_width, fmeasure);

  //  hmeasure = ((1-param.beta) * hfalse_reject + (param.beta)*hfalse_accept);
  double terr;
  if(optimize == OPT_FMEASURE)  terr = fmeasure; 
  //  else if(optimize == OPT_HINGE)  terr = hmeasure; 
  else if(optimize == OPT_RECALL)  terr = recall; 
  else if(optimize == OPT_BALANCEDRISK)  terr = risk; 
  else  terr = precision; 
  return terr;
}

struct openset_score_data{
  double label;
  double score;
};


int openset_compare_thresholds(const void *v1, const void *v2){
  double diff =(*(double*) v1) - (*(double*) v2);
  if(diff== 0) return 0;
  else     if(diff <  0) return -1;
  return 1;
}


int openset_compare_scores(const void *v1, const void *v2){
  double diff =((struct openset_score_data*) v1)->score - ((struct openset_score_data*) v2)->score;
  if(diff== 0) return 0;
  else     if(diff <  0) return -1;
  return 1;
}





void openset_find_planes(const struct svm_problem &prob,  struct svm_model *model,
                              struct openset_score_data * scores,    double *threshold,
                              double *alpha_ptr,double *omega_ptr, int correct_label){

    double maxval, minval;
    double min_error = 0, alpha = 0, omega = 0, width_at_min=.99999999e299;
    int base_min_index = -1;
    int base_max_index = -1;
    

    //if(optimize == OPT_HINGE || optimize == OPT_BALANCEDRISK )  
    min_error = .9999999e99; 
    max_line_len = 1024;
    //    line = (char *)malloc((ulong)max_line_len*sizeof(char));

	
    qsort(threshold,(size_t)prob.l,sizeof(double),openset_compare_thresholds);
    qsort(scores,(size_t)prob.l,sizeof(struct openset_score_data),openset_compare_scores);
    if(model->param.vfile)     fprintf(model->param.vfile,"Min %lf   Max %lf\n", threshold[0],threshold[prob.l-1]);
    int *rawnegcnt = (int *) malloc((ulong)(prob.l+3)*sizeof(int));
    int *rawposcnt = (int *) malloc((ulong)(prob.l+3)*sizeof(int));
    int *negcnt = rawnegcnt+1; // allow -1 index 
    int *poscnt = rawposcnt+1; // allow -1 index 

    
    int zindex=-1, mindex=-1, inclass=0;
    maxval= scores[0].score;
    minval = scores[prob.l-1].score;
    rawnegcnt[0] =     rawposcnt[0] =     negcnt[0] = poscnt[0] =0;
    for (int i = 0; i < prob.l; i++){
        if (scores[i].label == correct_label) { /* if a positive */ 
          poscnt[i+1] = poscnt[i] +1; 
          negcnt[i+1] = negcnt[i]; 
          ++inclass;
          /* find largest score for correct class */
          if(scores[i].score >= maxval) {
            maxval = scores[i].score;
            zindex = i;
          }
          /* find largest negative score for correct class */
          if(scores[i].score < minval) {
            minval = scores[i].score;
            mindex = i;
          }
          /* maybe base_max_index should be last pos poitn for a pos point  FIXME ??? */ 
          base_max_index=i+1; 

        } else { /* its a negatives */ 
          poscnt[i+1] = poscnt[i]; 
          negcnt[i+1] = negcnt[i]+1; 
          /* want to find the last (largest) negative score for a negative point  */ 
          if(scores[i].score < 0 && base_max_index != i) {
            base_min_index = i;
          } else {
            /* find last postive score for a negative point.. */
            //              base_max_index=i+1; 
          }
        }
    }

    // allow for index above prob.l
    for (int i = prob.l; i < prob.l+2; i++){
      poscnt[i] = poscnt[prob.l-1];
      negcnt[i] = negcnt[prob.l-1];
    }

    if(base_max_index<0 || base_max_index> prob.l-1 || model->param.kernel_type != LINEAR ) base_max_index = prob.l-1;
    /* compute max dilation for near and far planes */

    /* 
       Initally compute risk for max dilation of both near and far. 
       while not at a minum, compute improvment from moving near and far, move the one with greater improvement.
    */

    double precision = 0, recall = 0, fmeasure = 0, error = 0;
    int retrieved = 0, relevant = 0, correct = 0;
    precision = 0; recall = 0; fmeasure = 0; error = 0;
    int false_pos = 0, false_neg = 0;
    double hfalse_accept = 0, hfalse_reject = 0;
    int  true_neg=0, true_pos=0;
    

    int nearindex = base_min_index;
    if(nearindex <0) nearindex=0;
    int farindex = base_max_index;
    if(farindex < 0) farindex = prob.l-1;
    if(nearindex == farindex) nearindex=0;

#if 0
    // if there are negatives beyond the max positive, we set plane half wan between
    if(zindex < prob.l){
      double mv1 =   0;
      if(zindex < prob.l-1)  mv1 = (scores[zindex].score + scores[zindex+1].score)/2;
      double mv2 =   scores[zindex].score + fabs(scores[zindex].score - scores[zindex-1].score); 
      if(mv2 > mv1 && mv1 >0 ) maxval = mv1;
      else maxval = mv2;
    }
    



    if(nearindex <0) nearindex=0;

    /* using max dilation from paper */ 
    /* max dilation for near is half the distance to next largest score */ 
    double mv1 =   (scores[nearindex].score + scores[nearindex+1].score)/2; 
    while(nearindex > 0 && (scores[base_min_index].score - scores[nearindex-1].score) < mv1) nearindex--;
    if(nearindex <0) nearindex=0;

    if(farindex < 0) farindex = prob.l-1;

    /* max dilation for far plane is half the distance to next smallest score */ 
    mv1 =   (scores[farindex].score + scores[farindex-1].score)/2; 
    while(farindex >nearindex && farindex < prob.l && (scores[farindex+1].score) - scores[base_max_index].score) farindex++;
    if(farindex > prob.l-1) farindex = prob.l-1;

    /* alternative start outside the base */
    nearindex = base_min_index;
    farindex = (base_max_index);

#endif


    if(model->param.kernel_type != LINEAR) {
      farindex = prob.l-1; // if we are doign RBF.. there is not reall upperbound, so make it larger
    }
    double tl = scores[nearindex].score;
    double tu = scores[farindex].score;
    double width=(tu-tl);
    double pos_width = fabs(1.0*maxval- minval) ; /* width the all positive data down to 0 plane (i.e. one-class starting point)*/

    
    //Compute precision and recall
    for (int i = 0; i< prob.l ; i++){
      //|| param.kernel_type == RBF)
      if (scores[i].score >= tl && (scores[i].score <= tu)) {
          retrieved++;
          if (scores[i].label == correct_label) // a positive
          {
              relevant++;
              ++correct;
              true_pos++;
            }
          else{
            double tmp = min(fabs(scores[i].score - tl), fabs(tu - scores[i].score));
            hfalse_accept += max(width,tmp);
            false_pos++;
          }
          
        }
      else{
          if (scores[i].label == correct_label){
            double tmp = min(fabs(scores[i].score - tl), fabs(tu - scores[i].score));
            hfalse_reject += max(width,tmp); 
            false_neg++;
          }
          else{
            ++correct;
            true_neg++;
          }
        }
    }
          
    if (retrieved > 0)
      precision = ((double) relevant)/retrieved;
    else
      precision = 0;
    
    if(inclass!=0)
      recall = ((double) relevant)/inclass;
    else recall=0;

    int tp = poscnt[farindex+1]-poscnt[nearindex];
    int fp = negcnt[farindex+1]-negcnt[nearindex];
    int tn = negcnt[prob.l]-fp;
    int fn = poscnt[prob.l]-tp;


    min_error = openset_error_for_optimization(tu,tl,pos_width, tp, tn, fp, fn,model->param);
    width_at_min = width;
    alpha = tl; 
    omega = tu;

    precision=0; recall=0;
    if ((true_pos+false_pos) > 0)
      precision = ((double) (true_pos)/(true_pos+false_pos));
    if((true_pos + false_neg) > 0)
      recall = ((double) true_pos)/(true_pos + false_neg);

    double beta = model->param.beta;
    if(beta*precision + recall > 0) 
      fmeasure = (1+beta*beta) *precision*recall/(beta*beta*precision + recall);
    else fmeasure=0;
    double risk = openset_compute_risk(tu,tl, pos_width, fmeasure);

    if(model->param.vfile)    
      fprintf(model->param.vfile,"Before optimization min: %g, max: %g, precision: %g recall: %g F %g risk %g, error %g\n", 
              alpha, omega, precision, recall,fmeasure, risk,  min_error);



 
    if( ! model->param.exaustive_open)     {

      /*  Greed optimization does not alwaays really improve the results much but is much faster */      
      /* at this point we have risk, precision and recall.. now we optimize.  
         By choice of starting points we have max recall, so must decide if moving near or far improves things the most  */  
      int dtp[4], dtn[4], dfp[4],dfn[4];
      double near_err_plus=0,  near_err=0;
      double  far_err_plus=0, far_err=0;
      int tindex;

    
      memset(dtp,0,sizeof(dtp));
      memset(dtn,0,sizeof(dtn));
      memset(dfp,0,sizeof(dfp));
      memset(dfn,0,sizeof(dfn));

      /* find best near for fixed far */ 
      near_err = min_error;
      tindex = nearindex;
      for(int i=nearindex; i>0 ; i--){

        tp = poscnt[farindex+1]-poscnt[i];
        fp = negcnt[farindex+1]-negcnt[i];
        tn = negcnt[prob.l]-fp;
        fn = poscnt[prob.l]-tp;
      
      
        near_err_plus = openset_error_for_optimization(scores[farindex].score, scores[i].score,pos_width, tp, tn, fp, fn,model->param);

        /* if new best so far, keep parms */
        if(near_err_plus < near_err ){
          near_err = near_err_plus;
          tindex = i;
        }
        if(nearindex < base_max_index-2) break; // limit how far we can go below first error point  fixme should this be a %? 
      }
      nearindex = tindex;
      min_error = near_err;


      if(model->param.kernel_type == LINEAR) {
        /* find best near for fixed far */ 
        far_err = min_error;
        tindex = base_max_index;
        for(int i=base_max_index; i< prob.l ; i++){

          tp = poscnt[i+1]-poscnt[nearindex];
          fp = negcnt[i+1]-negcnt[nearindex];
          tn = negcnt[prob.l]-fp;
          fn = poscnt[prob.l]-tp;
          far_err_plus = openset_error_for_optimization(scores[i].score, scores[nearindex].score,pos_width, tp, tn, fp, fn,model->param);


          /* if new best so far, keep parms */
          if(far_err_plus < far_err ){
            far_err = far_err_plus;
            tindex = i;
          }
        }
        farindex = tindex;
        min_error = far_err;
      }


      /* find best near for fixed far */ 
      near_err = min_error;
      tindex = nearindex;
      for(int i=nearindex; i>0 ; i--){

        tp = poscnt[farindex+1]-poscnt[i];
        fp = negcnt[farindex+1]-negcnt[i];
        tn = negcnt[prob.l]-fp;
        fn = poscnt[prob.l]-tp;
      
      
        near_err_plus = openset_error_for_optimization(scores[farindex].score, scores[i].score,pos_width, tp, tn, fp, fn,model->param);

        /* if new best so far, keep parms */
        if(near_err_plus < near_err ){
          near_err = near_err_plus;
          tindex = i;
        }
        if(nearindex < base_max_index-2) break; // limit how far we can go below first error point  fixme should this be a %? 
      }
      nearindex = tindex;
      min_error = near_err;


      alpha = scores[nearindex].score;
      omega = scores[farindex].score;

      tp = poscnt[farindex+1]-poscnt[nearindex];
      fp = negcnt[farindex+1]-negcnt[nearindex];
      tn = negcnt[prob.l]-fp;
      fn = poscnt[prob.l]-tp;

      tl = alpha;
      tu = omega;
      width=(tu-tl);
    
    
      if ((tp+fp) > 0)
        precision = ((double) (tp)/(tp+fp));

      if((tp+tn) > 0)    
        recall = ((double) tp)/(tp+fn);
      else recall=0;

      beta = model->param.beta;
      if(beta*precision + recall > 0) 
        fmeasure = (1+beta*beta) *precision*recall/(beta*beta*precision + recall);
      else fmeasure=0;
      risk = openset_compute_risk(tu,tl, pos_width, fmeasure);


      if(model->param.vfile) fprintf(model->param.vfile,"After stage one min: %g, max: %g, precision: %g recall: %g F %g risk %g, error %g\n", alpha, omega, precision, recall,fmeasure, risk,  min_error);


      double  near_err_neg=0,far_err_neg=0;
      int min_index=0, max_index=0,delta_near=0, delta_far=0 ;

      /* now we do greed optimization from that starting point */ 

      int tp,  fp,  tn,  fn; 


      if(nearindex > 0){
        int tp = poscnt[farindex+1]-poscnt[nearindex-1];
        int fp = negcnt[farindex+1]-negcnt[nearindex-1];
        int tn = negcnt[prob.l]-fp;
        int fn = poscnt[prob.l]-tp;
        near_err_neg = openset_error_for_optimization(scores[farindex].score, scores[nearindex-1].score,pos_width, tp, tn, fp, fn,model->param);
      } else near_err_neg = 9e99;

      if(nearindex+1 <prob.l){
        tp = poscnt[farindex+1]-poscnt[nearindex+1];
        fp = negcnt[farindex+1]-negcnt[nearindex+1];
        tn = negcnt[prob.l]-fp;
        fn = poscnt[prob.l]-tp;
        near_err_plus = openset_error_for_optimization(scores[farindex].score, scores[nearindex+1].score,pos_width, tp, tn, fp, fn,model->param);
      } else near_err_plus = 9e99;


      if(near_err_plus < near_err_neg && near_err_plus < 1e99 )
        {delta_near = +1; near_err = near_err_plus;}
      else  if (near_err_neg < 1e99) {
        delta_near = -1; near_err = near_err_neg;
      }
      else   {delta_near = 0; near_err = 9e99;}



      if(model->param.kernel_type == LINEAR) {
        if(farindex > 0){
          tp = poscnt[farindex+1-1]-poscnt[nearindex];
          fp = negcnt[farindex+1-1]-negcnt[nearindex];
          tn = negcnt[prob.l]-fp;
          fn = poscnt[prob.l]-tp;
          far_err_neg = openset_error_for_optimization(scores[farindex-1].score, scores[nearindex].score,pos_width, tp, tn, fp, fn,model->param);
        } else far_err_neg = 9e99;


        if(farindex+1 < prob.l){
          tp = poscnt[farindex+1+1]-poscnt[nearindex];
          fp = negcnt[farindex+1+1]-negcnt[nearindex];
          tn = negcnt[prob.l]-fp;
          fn = poscnt[prob.l]-tp;
          far_err_plus = openset_error_for_optimization(scores[farindex+1].score, scores[nearindex].score,pos_width, tp, tn, fp, fn,model->param);
        } else far_err_plus = 9e99;

        if(far_err_plus < far_err_neg && far_err_plus < 1e99 )
          {delta_far = +1; far_err = far_err_plus;}
        else  if (far_err_neg < 1e99) {
          delta_far = -1; far_err = far_err_neg;}
        else           {delta_far = 0; far_err = 9e99;}
      } else far_err = near_err + 1; /* don't want to use the far side for non-linear kernel */ 

        
      while(near_err < min_error  || far_err < min_error ){
        //if near reduce more and we are not moving near to ofar(below first error point)  fixme should this be a %? 
        // this was an early limit (used in paper), removed while working on squrles
        //          if(near_err < far_err && nearindex >  base_min_index-5){/* if near reduced error more */ 
        if(near_err < far_err){/* if near reduced error more */ 
          /* don't let score regions overlap */ 
          if( scores[nearindex+delta_near].score >= scores[farindex].score) break;
          min_error = near_err;
          nearindex += delta_near;
          if(nearindex >=farindex) nearindex=farindex-1;;
          if(nearindex <1) nearindex=1;
          min_index=nearindex;
          alpha = scores[nearindex].score;
        } else if(far_err < min_error)  { /* far reduced error more */

          /* don't let score regions overlap */ 
          if( scores[nearindex].score >= scores[farindex+delta_far].score) break;

          min_error = far_err;
          farindex += delta_far;
          if(farindex >= prob.l) farindex = prob.l-1;
          if(farindex <= nearindex) farindex = nearindex+1;
          max_index=farindex;

          omega = scores[farindex].score;
        } else{ // reached base_min so don't go any farther
          min_error = near_err;
          nearindex += delta_near;
          if(nearindex >=farindex) nearindex=farindex-1;;
          if(nearindex <1) nearindex=1;
          min_index=nearindex;
          alpha = scores[nearindex].score;
          break;   
        }

        if(nearindex > 0){
          tp = poscnt[farindex+1]-poscnt[nearindex-1];
          fp = negcnt[farindex+1]-negcnt[nearindex-1];
          tn = negcnt[prob.l]-fp;
          fn = poscnt[prob.l]-tp;
          near_err_neg = openset_error_for_optimization(scores[farindex].score, scores[nearindex-1].score,pos_width, tp, tn, fp, fn,model->param);
        } else near_err_neg = 9e99;
          
        if(nearindex+1 < prob.l){          
          tp = poscnt[farindex+1]-poscnt[nearindex+1];
          fp = negcnt[farindex+1]-negcnt[nearindex+1];
          tn = negcnt[prob.l]-fp;
          fn = poscnt[prob.l]-tp;
          near_err_plus = openset_error_for_optimization(scores[farindex].score, scores[nearindex+1].score,pos_width, tp, tn, fp, fn,model->param);
        }else near_err_plus = 9e99;

        if(near_err_plus < near_err_neg && near_err_plus < 1e99 )
          {delta_near = +1; near_err = near_err_plus;}
        else  if (near_err_neg < 1e99) {
          delta_near = -1; near_err = near_err_neg;}
        else{ delta_near = 0; near_err = 9e99;}

        if(model->param.kernel_type == LINEAR) {

          if(farindex > 0) {
            tp = poscnt[farindex+1-1]-poscnt[nearindex];
            fp = negcnt[farindex+1-1]-negcnt[nearindex];
            tn = negcnt[prob.l]-fp;
            fn = poscnt[prob.l]-tp;
            far_err_neg = openset_error_for_optimization(scores[farindex-1].score, scores[nearindex].score,pos_width, tp, tn, fp, fn,model->param);
          } else far_err_neg = 9e99;
            
          if(farindex+1 < prob.l){
            tp = poscnt[farindex+1+1]-poscnt[nearindex];
            fp = negcnt[farindex+1+1]-negcnt[nearindex];
            tn = negcnt[prob.l]-fp;
            fn = poscnt[prob.l]-tp;
            far_err_plus = openset_error_for_optimization(scores[farindex+1].score, scores[nearindex].score,pos_width, tp, tn, fp, fn,model->param);
          } else far_err_plus = 9e99;
                      
          if(far_err_plus < far_err_neg && far_err_plus < 1e99 )
            {delta_far = +1; far_err = far_err_plus;}
          else  if (far_err_neg < 1e99) {
            delta_far = -1; far_err = far_err_neg;}
          else{ delta_far = 0; far_err = 9e99;}

        } else far_err = near_err + 1; /* don't want to use the far side for non-linear kernel */ 
      }
    }  else  { /* exaustive optimization */


      int bestfar, bestnear;
      double besterr;
      
      besterr = min_error;
      bestfar = farindex;
      bestnear = nearindex;
      
      int farstart=1;
      int farend = prob.l;
      if(model->param.kernel_type != LINEAR) {
        farstart=prob.l-1;
        farend=prob.l;
      }
            
      for(farindex=farstart; farindex<farend ; farindex++){
        while(farindex<(farend-1) && poscnt[farindex] == poscnt[farindex+1]) farindex++;
        for(nearindex=farindex-1; nearindex>0 ; nearindex--){
          while(nearindex>1 && poscnt[nearindex] == poscnt[nearindex-1]) nearindex--;
          tl = scores[nearindex].score;
          tu = scores[farindex].score;
          width=(tu-tl);
          if(fabs(width)< 1e-10) break;
          tp = poscnt[farindex+1]-poscnt[nearindex];
          fp = negcnt[farindex+1]-negcnt[nearindex];
          tn = negcnt[prob.l]-fp;
          fn = poscnt[prob.l]-tp;
    
        if ((tp+fp) > 0)
          precision = ((double) (tp)/(tp+fp));

        if((tp+tn) > 0)    
          recall = ((double) tp)/(tp+fn);
        else recall=0;
        beta = model->param.beta;
        if(beta*precision + recall > 0) 
          fmeasure = (1+beta*beta) *precision*recall/(beta*beta*precision + recall);
        else fmeasure=0;
        double err = openset_compute_risk(tu,tl, pos_width, fmeasure);
        //        fprintf(stderr,"doing optimization near: %d, far: %d, precision: %g recall: %g F %g risk %g, best %g\n", nearindex, farindex, precision, recall,fmeasure, err,  besterr);
        

        /* if new best so far, keep parms */
        if(err <= besterr ){
          besterr = err;
          bestfar = farindex;
          bestnear = nearindex;
        }
        }
      }

      nearindex = bestnear;
      farindex = bestfar;
      min_error = besterr;

    }/* end if doing optimization */

    /* done core optization, set plane variable (alpha omega)  and compute final risk, then see if we need half-step tweeks */ 
    alpha = scores[nearindex].score;
    omega = scores[farindex].score;

    tp = poscnt[farindex+1]-poscnt[nearindex];
    fp = negcnt[farindex+1]-negcnt[nearindex];
    tn = negcnt[prob.l]-fp;
    fn = poscnt[prob.l]-tp;

    tl = alpha;
    tu = omega;
    width=(tu-tl);
    
    
    if ((tp+fp) > 0)
      precision = ((double) (tp)/(tp+fp));

    if((tp+tn) > 0)    
      recall = ((double) tp)/(tp+fn);
    else recall=0;

    beta = model->param.beta;
    if(beta*precision + recall > 0) 
      fmeasure = (1+beta*beta) *precision*recall/(beta*beta*precision + recall);
    else fmeasure=0;
    risk = openset_compute_risk(tu,tl, pos_width, fmeasure);

    if(model->param.exaustive_open &&  model->param.vfile ) fprintf(model->param.vfile,"After stage one min: %g, max: %g, precision: %g recall: %g F %g risk %g, error %g\n", alpha, omega, precision, recall,fmeasure, risk,  min_error);

    /* if we have mixed answers on end, we we take point  between based on generalization preasure*/
    if(nearindex > 0 && scores[nearindex].label == correct_label  && scores[nearindex-1].label != correct_label){
      //      alpha = (scores[nearindex].score * (1-near_preasure/2) + scores[nearindex-1].score*(near_preasure/2));
      alpha = (scores[nearindex].score * (1-model->param.near_preasure/2) + scores[nearindex-1].score*(model->param.near_preasure/2)); 
      if(model->param.vfile) fprintf(model->param.vfile,"Near small margin adjust, %lf\n" , scores[nearindex].score -alpha );

    }
    // if first point, and last label was positive, expand down a bit using spacing from next to last point 
    else if(nearindex == 0 && scores[nearindex].label == correct_label  ) {
      alpha = scores[nearindex].score - pos_width * (model->param.near_preasure/2);
      if(model->param.vfile) fprintf(model->param.vfile,"Near large margin adjust, %lf\n" , scores[nearindex].score -alpha );
    }

    if(scores[farindex].score < maxval && farindex+1 < prob.l && scores[farindex].label == correct_label  && scores[farindex+1].label != correct_label)  {
      //      omega = (scores[farindex].score* (1-model->param.far_preasure/2)) + scores[farindex+1].score * (model->param.far_preasure/2);
      omega = scores[farindex].score +  pos_width * (model->param.far_preasure/2);
      if(model->param.vfile) fprintf(model->param.vfile,
                                     "Far small large margin adjust, %lf\n" , 
                                     (omega - scores[farindex].score ));
    }
    // if last point, and last label was positive, expand up a bit using spacing from next to last point 
    else if(farindex+1 == prob.l){
      omega = scores[farindex].score + pos_width * (model->param.far_preasure/2);
      if(model->param.vfile) fprintf(model->param.vfile,
                                     "Far large margin adjust, %lf\n" , 
                                     (omega - scores[farindex].score) );
    }
    if(model->param.kernel_type != LINEAR) omega += 10; /* for RBF no real max.. make it a bit larger to address noise */ 

    tp = poscnt[farindex+1]-poscnt[nearindex];
    fp = negcnt[farindex+1]-negcnt[nearindex];
    if(farindex==prob.l){
      tp = poscnt[farindex]-poscnt[nearindex];
      fp = negcnt[farindex]-negcnt[nearindex];
    }
    tn = negcnt[prob.l]-fp;
    fn = poscnt[prob.l]-tp;

    tl = alpha;
    tu = omega;
    width=(tu-tl);
    
    if ((tp+fp) > 0)
      precision = ((double) (tp)/(tp+fp));
    else precision=0;

    if((tp+tn) > 0)    
      recall = ((double) tp)/(tp+fn);
    else recall=0;    

    //    if(recall != 0 )  risk =  (width/pos_width + pos_width/width)/2 + param.beta /(precision*recall/(precision + recall));
    beta = model->param.beta;
    if(beta*precision + recall > 0) 
      fmeasure = (1+beta*beta) *precision*recall/(beta*beta*precision + recall);
    else fmeasure=0;
    risk = openset_compute_risk(tu,tl, pos_width, fmeasure);
    if(model->param.vfile) fprintf(model->param.vfile,"After Stage 2  min: %g, max: %g, precision: %g recall: %g F %g risk %g, error %g\n\n", alpha, omega, precision, recall,fmeasure, risk,  min_error);


    if (model->param.optimize == OPT_PRECISION){
      if(model->param.vfile) fprintf(model->param.vfile,"Optimizing precision given recall with beta of %g\n", model->param.beta);
    }
    else     if (model->param.optimize == OPT_RECALL){
      if(model->param.vfile)         fprintf(model->param.vfile,"Optimizing recall given precision with beta of of %g\n", model->param.beta);
    }
    else     if (model->param.optimize == OPT_FMEASURE){
      if(model->param.vfile)         fprintf(model->param.vfile,"Optimizing Fmeasure \n");
    }

    if(model->param.vfile)         fflush(model->param.vfile);
    if(rawnegcnt) free(rawnegcnt); 
    if(rawposcnt) free(rawposcnt); 

    // push the actual answers back through the arguments
    *alpha_ptr = alpha;
    *omega_ptr = omega;
}

void openset_analyze_set(const struct svm_problem &prob,  struct svm_model *model,  double *alpha_ptr,double *omega_ptr, int correct_label){


    struct openset_score_data * scores = (struct openset_score_data *) malloc((ulong)prob.l*sizeof(struct openset_score_data));
    double *threshold  = (double *) malloc((ulong)(prob.l+2)*sizeof(double));

    int *vote = NULL;
    double **deepscores = Malloc(double *, model->nr_class);
    for(int v=0; v<model->nr_class; v++){
        deepscores[v] = Malloc(double, model->nr_class);
        for(int z=0; z<model->nr_class; z++)
          deepscores[v][z] = 0;
    }
    int nr_class = model->nr_class;
    double *dec_values = Malloc(double, (nr_class*(nr_class-1)/2));    
    

    for (int i = 0; i < prob.l; i++){
      scores[i].label = prob.y[i];
      svm_predict_values_extended(model,prob.x[i],dec_values, deepscores, vote);
      threshold[i] =   scores[i].score = dec_values[0];
    }

    
    if(vote != NULL)
      free(vote);

    if(dec_values != NULL)
      free(dec_values);
    
    for(int v=0; v<model->nr_class; v++)
      if(deepscores[v] != NULL)
        free(deepscores[v]);
    
    if(deepscores != NULL)
      free(deepscores);



    openset_find_planes(prob,  model, scores,    threshold,alpha_ptr,omega_ptr,correct_label);
    free(scores);
    free(threshold);
}

void openset_analyze_pairs(const struct svm_problem &prob,  struct svm_model *model){

    struct openset_score_data * scores = (struct openset_score_data *) malloc((ulong)prob.l*sizeof(struct openset_score_data));
    double *threshold  = (double *) malloc((ulong)(prob.l+2)*sizeof(double));

    int nr_class = model->nr_class;
    double ***rawscores = Malloc(double**, prob.l);
    for(int p=0; p<prob.l; p++){
      rawscores[p] = Malloc(double*, nr_class);
      for(int i=0; i<nr_class; i++){
          rawscores[p][i]= Malloc(double, nr_class);
          memset(rawscores[p][i],0,nr_class*sizeof(double));
        }
    }
    
    int* votes = Malloc(int,nr_class);
    /* get scores for all training data.. for all classes */ 
    for (int p = 0; p < prob.l; p++){
      scores[p].label = prob.y[p];
      svm_predict_extended(model,prob.x[p], rawscores[p], votes); 
    }
    free(votes);


    /* to analyze pairs, we note there are n*(n-1)/2 different svms and score sets;  we process each separately */

    for(int fclass=0; fclass < nr_class; fclass++){
      for(int sclass=fclass+1; sclass < nr_class; sclass++){
        for (int i = 0; i < prob.l; i++){
          threshold[i] =   scores[i].score = rawscores[i][fclass][sclass];
        }
        double alpha,omega;
        openset_find_planes(prob,  model, scores,    threshold,&alpha,&omega,model->label[fclass]);
        model->alpha[fclass*nr_class+sclass] = alpha;
        model->omega[fclass*nr_class+sclass] = omega;
        
        for (int i = 0; i < prob.l; i++){
          threshold[i] =   scores[i].score = rawscores[i][sclass][fclass];
        }
        openset_find_planes(prob,  model, scores,    threshold,&alpha,&omega,model->label[sclass]);
        model->alpha[sclass*nr_class+fclass] = alpha;
        model->omega[sclass*nr_class+fclass] = omega;
      }
    }

    for(int i=0; i<prob.l; i++){
        for(int j=0; j<nr_class; j++)
          free(rawscores[i][j]);
        free(rawscores[i]);
      }
    free(rawscores);
   free(scores);
   free(threshold);
}

