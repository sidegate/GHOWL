#ifndef __CUDALIB_H__
#define __CUDALIB_H__

#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <functional>
#include <set>
#include <map>

using namespace std;

//////////////////////////////////////////////////////////////////////////
// Short typedefs and forward declares
//////////////////////////////////////////////////////////////////////////
typedef unsigned int dword;
typedef unsigned char byte;
typedef unsigned short word;
typedef char *pchar;
typedef const char *pcchar;
typedef void** ppv;


template<typename T> struct TXelCoOrd;
template<typename T> struct TPelCoOrd;
template<typename T> struct TCoOrd3D;

typedef TPelCoOrd<int> TXY;
typedef TPelCoOrd<double> TFXY;

typedef TXelCoOrd<int> TIJ;
typedef TXelCoOrd<double> TFIJ;

typedef TCoOrd3D<int> TXYZ;
typedef TCoOrd3D<double> TFXYZ;

template<int N, class T> struct TMany;

// CUDA Emulation stuff
#include "cudaemu.h"

///////////////////////////////////////////////////////////////////////////
/// Macros for looping over 2 or 3 dimensions
///////////////////////////////////////////////////////////////////////////
#define LOOP_YX(W, H)               \
    for(dword y = 0; y < H; ++y)    \
    for(dword x = 0; x < W; ++x)

#define LOOP_XY(W, H)           \
    for(dword x = 0; x < W; ++x) \
    for(dword y = 0; y < H; ++y)  

#define LOOP_ZYX (W, H, D)        \
    for(dword z = 0; z < D; ++z)  \
    LOOP_YX(W, H)

///////////////////////////////////////////////////////////////////////////
/// Macros for frequently used trigonometric constants
///////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979323F
#define COS60 0.5F
#define SIN60 0.86602540378443864676372317075294F
#define SIN30 COS60
#define COS30 SIN60
#define TAN30 (SIN30/COS30)
#define COT30 (COS30/SIN30)
#define TAN60 (SIN60/COS60)
#define COT60 (COS60/SIN60)


//////////////////////////////////////////////////////////////////////////
/// CUDA kernel code does not manage to do copy construction so we need to roll
/// our own. 
//////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__
    #define DECL_CUDA_COPY_CTOR(X) \
        __GPU_CPU__ X(const X& other) { *this = other; } /// Use assignment operator to copy an object
#else
    #define DECL_CUDA_COPY_CTOR(X) ;
#endif

///////////////////////////////////////////////////////////////////////////
/// TCudaException converts CUDA error codes to C++ exceptions
/// Assigning an integer to a TCudaException object will cause it to 
/// get thrown
//////////////////////////////////////////////////////////////////////////
class TCudaException
{
    // Private copy constructor
    TCudaException(const TCudaException &)
    {}
    //////////////////////////////////////////////////////////////////////////

public:
    /// Default constructor does nothing
    TCudaException()
    {}
    //////////////////////////////////////////////////////////////////////////

    /// Assignment from integer ( typically a result from a CUDA API call )
    void operator=(int e)
    {
        if(e)
            throw e;
    }
    //////////////////////////////////////////////////////////////////////////

    static string errorMessage(int e)
    {
        static const char *sCudaErrorMessages[] = 
        {
            "No errors",
            "Missing configuration error",
            "Memory allocation error",
            "Initialization error",
            "Launch failure",
            "Prior launch failure",
            "Launch timeout error",
            "Launch out of resources error",
            "Invalid device function",
            "Invalid configuration",
            "Invalid device",
            "Invalid value",
            "Invalid pitch value",
            "Invalid symbol",
            "Map buffer object failed",
            "Unmap buffer object failed",
            "Invalid host pointer",
            "Invalid device pointer",
            "Invalid texture",
            "Invalid texture binding",
            "Invalid channel descriptor",
            "Invalid memcpy direction",
            "Address of constant error",
            "Texture fetch failed",
            "Texture not bound error",
            "Synchronization error",
            "Invalid filter setting",
            "Invalid norm setting",
            "Mixed device execution",
            "CUDA runtime unloading",
            "Unknown error condition",
            "Function not yet implemented",
            "Memory value too large",
            "Invalid resource handle",
            "Not ready error",
            "CUDA runtime is newer than driver",
            "Set on active process error",
            "No available CUDA device",
        };

        if(e >= 0 && e <= 38) 
            return sCudaErrorMessages[e];
        else if(e == 0x7f)
            return "Startup failure";
        else if(e >= 1000)
            return "API error";

        return "Unknown error";

    }
};
//////////////////////////////////////////////////////////////////////////

/// Global TCudaException variable which can be assigned to the return values of CUDA API calls
TCudaException g_e;
//////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
/// TXelCoOrd and TPelCoOrd represent PEL and XEL co ordinates
/// Co-ordinates are converted between XEL and PEL representations by assignment
/// TXelCoOrd3d and TPelCoOrd3d (TODO) are the 3d analogues
/////////////////////////////////////////////////////////////////////////

template<typename T> struct TPelCoOrd
{
    T x, y;
    typedef T Type;

    DECL_CUDA_COPY_CTOR(TPelCoOrd);

    // Default ctor
    __GPU_CPU__ TPelCoOrd() : x(0), y(0)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    // Construct from x and y
    __GPU_CPU__ TPelCoOrd(T ax, T ay) : x(ax), y(ay)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    template<typename T2> __GPU_CPU__ TPelCoOrd(const T2 &p)
    {
        *this = p;
    }
    //////////////////////////////////////////////////////////////////////////

    template<typename T2> __GPU_CPU__ TPelCoOrd& operator=(const T2 &p)
    {
        x = (T)p.x;
        y = (T)p.y;
        return *this;
    }
    //////////////////////////////////////////////////////////////////////////

    template<typename TOperand, class TOperator> 
    __GPU_CPU__ void op(const TOperand ox, const TOperand oy, TOperator &o)
    {
        x = o((TOperand)x, ox);
        y = o((TOperand)y, oy);
    }
    //////////////////////////////////////////////////////////////////////////

    bool operator <(const TPelCoOrd &o) const
    {
        return o.x == x ? y < o.y : x < o.x;
    }
    //////////////////////////////////////////////////////////////////////////

    bool operator==(const TPelCoOrd & o)
    {
        return o.x == x && o.y == y;
    }
    //////////////////////////////////////////////////////////////////////////

    // index helpers
    __GPU_CPU__ int index(int w, int h)
    {
        return h * y + x;
    }
    //////////////////////////////////////////////////////////////////////////

    __GPU_CPU__ void inc(int n, int w, int h)
    {
        int i = index(w, h) + n;
        x = i % w;
        y = i / h;
    }
};
//////////////////////////////////////////////////////////////////////////

template<typename T> struct TCoOrd3D
{
    T x, y, z;
    typedef T Type;

    DECL_CUDA_COPY_CTOR(TCoOrd3D);

    __GPU_CPU__ TCoOrd3D():x(0), y(0), z(0)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    template<typename T2> __GPU_CPU__ TCoOrd3D(T2 &p)
    {
        *this = p;
    }
    //////////////////////////////////////////////////////////////////////////

    template<typename T2> __GPU_CPU__ TCoOrd3D& operator=(const T2 &p)
    {
        x = (T)p.x;
        y = (T)p.y;
        z = (T)p.z;
        return *this;
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////

template<typename T> struct TXelCoOrd
{
    T i, j;
    typedef T Type;

    DECL_CUDA_COPY_CTOR(TXelCoOrd);

    __GPU_CPU__ TXelCoOrd() : i(0), j(0)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    template<typename T2> __GPU_CPU__ TXelCoOrd(T2 &p)
    {
        *this = p;
    }
    //////////////////////////////////////////////////////////////////////////

    template<typename T2> __GPU_CPU__ TXelCoOrd& operator=(T2 &p)
    {
        i = (T)p.i;
        j = (T)p.j;
        return *this;
    }
    //////////////////////////////////////////////////////////////////////////

    // Convert PEL Co-ords to XEL space
    template<typename T2> __GPU_CPU__ TXelCoOrd& operator=(const TPelCoOrd<T2> &p)
    {
        i = (T)(p.x / COS60);
        j = (T)(p.j * COS30);
        return *this;
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////

// Special class to help create arrays of structs in CUDA
// Beware of non-trivial constructors!!!
// Plain arrays of structs do not work in kernel code if the struct has a ctor
template<int N, class T> struct TMany
{
    byte pts[N * sizeof(T)];
    int size;  // convenient place to hold the size 

    __GPU_CPU__ TMany()
    {
        T a;
        for(int i = 0; i < N ; ++i)
            (*this)[i] = a;
    }

    __GPU_CPU__ T &operator[](int n)
    {
        return ((T*)pts)[n];
    }
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Helpers for [] operator support on pitched buffers
//////////////////////////////////////////////////////////////////////////

struct TPitchPtr
{
   char *ptr;
   int pitch;    ///< Pitch of allocated memory in bytes
   int w;    ///< Logical width of allocation in elements
   int h;    ///< Logical height of allocation in elements
   int d;
};
//////////////////////////////////////////////////////////////////////////

template<typename T> struct T2DArr : public TPitchPtr
{
    DECL_CUDA_COPY_CTOR(T2DArr);

    __GPU_CPU__ T2DArr(const TPitchPtr &p, int z = 0): TPitchPtr(p)
    {
        ptr += pitch * h * z;
    }

    __GPU_CPU__ T& operator[](const TXY &p) const
    {
        return ((T*)(ptr + pitch * p.y))[p.x];
    }

    __GPU_CPU__ T *operator[](int iy) const
    {
        return (T*)(ptr + iy * pitch);
    }
};
//////////////////////////////////////////////////////////////////////////

template<typename T> struct T3DArr : public TPitchPtr
{
    DECL_CUDA_COPY_CTOR(T3DArr);

    __GPU_CPU__ T3DArr(const TPitchPtr &p) : TPitchPtr(p)
    {
    }
    //////////////////////////////////////////////////////////////////////////

    __GPU_CPU__ T2DArr<T> operator[](int iz)
    {
        assert(iz > -1 && iz < d);
        return T2DArr<T>(*this, iz);
    }
    //////////////////////////////////////////////////////////////////////////

    __GPU_CPU__ T& operator[](const TXYZ &p)
    {
        assert(p.z > -1 &&  p.w > -1 && p.h > -1 && p.z < d && p.y < h && p.x < w);
        return T2DArr<T>(*this, p.z)[p.y][p.x];
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////

// XEL Matrix [] operator helper
template<typename T> struct T2DXArr : public TPitchPtr
{
    DECL_CUDA_COPY_CTOR(T2DXArr);

    __GPU_CPU__ T2DXArr(const TPitchPtr &p, int z): TPitchPtr(p)
    {
        ptr += pitch * h * z;
    }

//     __GPU_CPU__ T& operator[](const TIJ &p) const
//     {
//         assert(p.z > -1 &&  p.w > -1 && p.h > -1 && p.z < d && p.y < h && p.x < w);
//         return ((T*)(ptr + pitch * p.j))[p.i + p.j / 2];
//     }

    __GPU_CPU__ T *operator[](int iy) const
    {
        assert(iy > -1 && iy < h);
        return ((T*)(ptr + iy * pitch)) + iy / 2;
    }
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Base class for host and buffer wrapper classes
//////////////////////////////////////////////////////////////////////////
template<typename T> class TArray
{
protected:

    T* m_pBuf;              // Pointer to storage
    dword m_cbPitch;        // Number of bytes per row
    cudaExtent m_dim;       // dimensions of this array

public:

    // size of the element stored
    static const int Size = sizeof(T);
    typedef T Elem;

    // Initially array has no storage and no dimensions
    TArray()
    {
        m_pBuf = 0;
        m_cbPitch = m_dim.width = m_dim.height = m_dim.depth = 0;
    }
    //////////////////////////////////////////////////////////////////////////

    // Construct 1d, 2d or 3d array
    TArray(int ix, int iy = 1, int iz = 1)
    {
        m_dim.width = ix;
        m_dim.height = iy;
        m_dim.depth = iz;
        m_cbPitch = ix * Size;
        m_pBuf = 0;
    }
    //////////////////////////////////////////////////////////////////////////

    virtual ~TArray(){}
    //////////////////////////////////////////////////////////////////////////

    // Compares 2 arrays' dimensions and throws if unequal
    void checkDims(const TArray<T> &that) const
    {
        const cudaExtent &e = that.dims();
        if(depth() != e.depth || height() != e.height || width() != e.width)
            throw string("Dimensions mismatch between buffers");
    }
    //////////////////////////////////////////////////////////////////////////

    // Required only internally for 2d memcpys
    // (has to be public since another class needs access)
    inline cudaPitchedPtr pitchPtr() const
    {
        cudaPitchedPtr p;
        p.pitch = m_cbPitch;
        p.ptr = m_pBuf;
        p.xsize = m_dim.width;
        p.ysize = m_dim.height;
        return p;
    }
    //////////////////////////////////////////////////////////////////////////

    TPitchPtr data()
    {
        TPitchPtr p;
        p.pitch = m_cbPitch;
        p.w= m_dim.width;
        p.h= m_dim.height;
        p.d = m_dim.depth;
        p.ptr = (char*)m_pBuf;
        return p;
    }
    //////////////////////////////////////////////////////////////////////////

    inline T *ptr() const 
    {
        return m_pBuf;
    }
    //////////////////////////////////////////////////////////////////////////

    inline dword width() const
    {
        return  m_dim.width;
    }
    //////////////////////////////////////////////////////////////////////////

    inline dword height() const
    {
        return  m_dim.height;
    }
    //////////////////////////////////////////////////////////////////////////

    inline dword depth() const
    {
        return  m_dim.depth;
    }
    //////////////////////////////////////////////////////////////////////////

    inline int pitch() const
    {
        return m_cbPitch;
    }
    //////////////////////////////////////////////////////////////////////////

    inline const cudaExtent &dims() const
    {
        return m_dim;
    }
    //////////////////////////////////////////////////////////////////////////
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Helper functor for memory copies Device->Device Host->Device and vice versa
//////////////////////////////////////////////////////////////////////////
template<typename T_TO, typename T_FROM, cudaMemcpyKind kind> struct TArrayCopier
{
    void operator()(const T_TO &to, const T_FROM &from)
    {
        to.checkDims(from);

        // Check the geometry of the device buffer 
        if(from.depth() > 1) // 3d
        {
            cudaMemcpy3DParms p = {0};
            p.srcPtr = from.pitchPtr();
            p.dstPtr = to.pitchPtr();
            p.extent = from.dims();
            p.extent.width *= to.Size;
            p.kind = kind;

            g_e = cudaMemcpy3D(&p);

        }
        else if(from.height() > 1)
        {
            g_e = cudaMemcpy2D(to.ptr(), to.pitch(), from.ptr(), from.pitch(), 
                to.width() * to.Size, to.height(), kind);
        }
        else
        {
            g_e = cudaMemcpy(to.ptr(), from.ptr(), from.pitch(), kind);
        }
    }
};
//////////////////////////////////////////////////////////////////////////

// Forward declare
template<typename T> class TGPUArray;
template<typename T> class TCPUArray;
template<typename T> class TBitmap;

//////////////////////////////////////////////////////////////////////////
// Host buffer wrapper class
//////////////////////////////////////////////////////////////////////////
template<typename T> class TCPUArray : public TArray<T>
{
protected:

    typedef TArrayCopier<TCPUArray, TGPUArray<T>, cudaMemcpyDeviceToHost> 
        TCopyDeviceToHost;

    vector<char> m_v;

    void resize()
    {
        int ix = this->m_dim.width;
        int iy = this->m_dim.height;
        int iz = this->m_dim.depth;

        // cant have 0 for any dimension
        assert(ix && iy && iz);

        // Align on 16 byte boundary
        this->m_cbPitch = ix * this->Size;
        if(this->m_cbPitch % 16)
        {
            this->m_cbPitch &= ~16;
            this->m_cbPitch += 16;
        }

        // Save old size
        int iOldSize = m_v.size() / this->Size;

        // Resize the array and save the pointer
        m_v.resize(this->m_cbPitch * iy * iz);
        
        //fill(m_v.begin(), m_v.end(), 0);
        this->m_pBuf = (T*)(&*m_v.begin());

        // Construct objects in the empty area if size grew
        int nNew = (ix * iy * iz) - iOldSize;
        T* p = this->m_pBuf;
        while(nNew --> 0)
        {
            new(p) T();
            p++;
        }

    }
    //////////////////////////////////////////////////////////////////////////

public:

    const static bool IsDevice = false;

    TCPUArray() : TArray<T>(0, 0, 0)
    {
    }

    // Construct given dimensions
    TCPUArray(int ix, int iy = 1, int iz = 1) : TArray<T>(ix, iy, iz)
    {
        resize();
    }
    //////////////////////////////////////////////////////////////////////////

    TCPUArray(const cudaExtent &e) : TArray<T>(e.width, e.height, e.depth)
    {
        resize();
    }
    //////////////////////////////////////////////////////////////////////////

    // Copy ctor has to be defined even though we have the template one
    TCPUArray(const TCPUArray &that) : 
        TArray<T>(that.width(), that.height(), that.depth())
    {
        resize();
        *this = that;
    }
    //////////////////////////////////////////////////////////////////////////

    // Construct from any sibling class for which we have a = operator
    // (dimensions and data will get imbued)
    template<class T2> TCPUArray(const T2 &that) : 
        TArray<T>(that.width(), that.height(), that.depth())
    {
        resize();
        TArrayCopier<TCPUArray, T2, 
            T2::IsDevice ? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost>
            ()(*this, that);
    }

    // Assign data from another host buffer, works to assign to blank buffer too
    void operator=(const TCPUArray &that)
    {
        if(!m_pBuf)
        {
            m_dim = that.m_dim;
            m_cbPitch = that.m_cbPitch;
        }

        checkDims(that);
        m_v = that.m_v;
        this->m_pBuf = (T*)(&m_v[0]);
    }
    //////////////////////////////////////////////////////////////////////////

    // Assignment from device buffer
    void operator=(TGPUArray<T> &that)
    {
        TCopyDeviceToHost()(*this, that);
    }
};
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Device buffer wrapper class
//////////////////////////////////////////////////////////////////////////
template<typename T> class TGPUArray : public TArray<T>
{
    friend class TCPUArray<T>;

    typedef TArrayCopier<TGPUArray, TGPUArray, cudaMemcpyDeviceToDevice> 
        TCopyDeviceToDevice;

    typedef TArrayCopier<TGPUArray, TCPUArray<T>, cudaMemcpyHostToDevice> 
        TCopyHostToDevice;


    // (re)allocates CUDA memory 
    void mallocate()
    {
        if(this->m_pBuf)
        {
            g_e = cudaFree(this->m_pBuf);
            this->m_pBuf = 0;
        }

        int ix = this->m_dim.width;
        int iy = this->m_dim.height;
        int iz = this->m_dim.depth;

        if(iz == 1)
        {
            if(iy == 1) // 1D
            {
                T *p;
                g_e = cudaMalloc(ppv(&p), ix * iy * iz * this->Size);
                this->m_pBuf = p;
            }
            else
            {
                dword uPitch;
                void *pBuf = this->m_pBuf;
                g_e = cudaMallocPitch(&pBuf, &uPitch, ix * this->Size, iy);
                this->m_cbPitch = uPitch;
                this->m_pBuf = (T*)pBuf;
            }
        }
        else // 3D
        {
            cudaPitchedPtr p;
            cudaExtent dim = this->m_dim;
            dim.width *= this->Size;
            g_e = cudaMalloc3D(&p, dim);
            this->m_cbPitch = p.pitch;
            this->m_pBuf = (T*)p.ptr;
        }
    }
    //////////////////////////////////////////////////////////////////////////

public:

    const static bool IsDevice = true;

    // Construct 1 2 or 3D buffer given dimensions
    TGPUArray(int ix, int iy = 1, int iz = 1) : TArray<T>(ix, iy, iz)
    {
        mallocate();
    }
    //////////////////////////////////////////////////////////////////////////

    TGPUArray(const cudaExtent &e) : TArray<T>(e.width, e.height, e.depth)
    {
        mallocate();
    }
    //////////////////////////////////////////////////////////////////////////

    // Copy ctor has to be defined even though we have the template one
    TGPUArray(const TGPUArray &that) : TArray<T>(that.width(), that.height(), that.depth())
    {
        mallocate();
        TCopyDeviceToDevice()(*this, that);
    }
    //////////////////////////////////////////////////////////////////////////

    // Default ctor, with 0 storage
    TGPUArray()
    {
    }
    //////////////////////////////////////////////////////////////////////////

    // Free allocated memory 
    virtual ~TGPUArray()
    {
        cudaFree(this->m_pBuf);
    }
    //////////////////////////////////////////////////////////////////////////

    // Construct from any sibling class for which we have a = operator
    // (dimensions and data will get imbued)
    template<class T2> TGPUArray(const T2 &that) : 
        TArray<T>(that.width(), that.height(), that.depth())
    {
        mallocate();
        
        TArrayCopier<TGPUArray, T2, 
            T2::IsDevice ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice>
            ()(*this, that);
    }
    //////////////////////////////////////////////////////////////////////////

    // Assign from host memory
    void operator=(const TCPUArray<T> &that)
    {
        TCopyHostToDevice()(*this, that);
    }
    //////////////////////////////////////////////////////////////////////////

    // Assign from device memory
    void operator=(TGPUArray<T> &that)
    {
        TCopyDeviceToDevice()(*this, that);
    }
    //////////////////////////////////////////////////////////////////////////

};
//////////////////////////////////////////////////////////////////////////


// Get the 4 box co-ordinates that enclose a pel co-ordinate
__GPU_CPU__ void getBoxBounds(TFXY pt, TMany<4, TXY> &bounds)
{
    bounds[0].x = (int)pt.x;
    bounds[0].y = (int)pt.y;
    bounds[1].x = (int)pt.x;
    bounds[1].y = (int)pt.y - 1;
    bounds[2].x = (int)pt.x + 1;
    bounds[2].y = (int)pt.y - 1;
    bounds[3].x = (int)pt.x + 1;
    bounds[3].y = (int)pt.y;
}
//////////////////////////////////////////////////////////////////////////

// Get the 6 neighboring xels for a given one ( in box co ords )
__GPU_CPU__ void getNearXels(TXY &xel, TXY nears[6])
{
    int iIsOdd = xel.y % 2;
    int iIsEven = 1 - iIsOdd;

    nears[0].x = xel.x + 1;
    nears[0].y = xel.y;

    nears[1].x = xel.x + iIsOdd;
    nears[1].y = xel.y + 1;

    nears[2].x = xel.x - iIsEven;
    nears[2].y = xel.y + 1;

    nears[3].x = xel.x - 1;
    nears[3].y = xel.y;

    nears[4].x = xel.x - iIsEven;
    nears[4].y = xel.y - 1;

    nears[5].x = xel.x + iIsOdd;
    nears[5].y = xel.y - 1;
}
//////////////////////////////////////////////////////////////////////////

// Get the 8 neighboring pels for a given one
__GPU_CPU__ void getNearPels(TXY &pel, TXY nears[8])
{
    // Left and right
    nears[0].x = pel.x - 1;
    nears[0].y = pel.y;

    nears[4].x = pel.x + 1;
    nears[4].y = pel.y;

    // Upper 3
    nears[1].x = pel.x - 1;
    nears[1].y = pel.y + 1;

    nears[2].x = pel.x;
    nears[2].y = pel.y + 1;

    nears[3].x = pel.x + 1;
    nears[3].y = pel.y + 1;

    // Lower 3
    nears[5].x = pel.x + 1;
    nears[5].y = pel.y - 1;

    nears[6].x = pel.x;
    nears[6].y = pel.y - 1;

    nears[7].x = pel.x - 1;
    nears[7].y = pel.y - 1;
}
//////////////////////////////////////////////////////////////////////////

// Interpolate the value at fDist if the value at 0 is f1 and at 1 is f2
__GPU_CPU__ double interpolate(double fDist, float f1, double f2)
{
    return f1 + ((f2 - f1) * fDist);
}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// CUDA thread helpers
//////////////////////////////////////////////////////////////////////////

// Get the absolute linear identity of a thread
__device__ int getLinearThreadIndex()
{
    int nBlock =
        blockIdx.z * (gridDim.x * gridDim.y) +
        blockIdx.y * (gridDim.x) +
        blockIdx.x;

    int nThread =
        nBlock * (blockDim.x * blockDim.y * blockDim.z) +
        threadIdx.z * (blockDim.x * blockDim.y) +
        threadIdx.y * (blockDim.x) +
        threadIdx.x;

    return nThread;
}
//////////////////////////////////////////////////////////////////////////

__device__ int getThreadX(int nMax)
{
    return getLinearThreadIndex() % nMax;
}
//////////////////////////////////////////////////////////////////////////
 
// Get the virtual X and Y IDs of a thread given the virtual width and height
__device__ TXY getThreadXY(int w, int h)
{
    TXY p;
    int nThread = getLinearThreadIndex();

    p.x = nThread % w;
    p.y = (nThread / w) % h;
    return p;
}
//////////////////////////////////////////////////////////////////////////

__device__ TXYZ getThreadXYZ(int w, int h, int d)
{
    TXYZ p;
    int nThread = getLinearThreadIndex();

    p.x = nThread % w;
    p.y = (nThread / w) % h;
    p.z = (nThread / (w * h) ) % d;
    return p;
}
//////////////////////////////////////////////////////////////////////////

void getBlockGridSizes(int nElems, int &nBlocks, int &nThreadsPerBlock)
{
    // Todo get max threads from cuda lib
    nThreadsPerBlock = 512;
    nBlocks = nElems / nThreadsPerBlock;
    if(nElems % nThreadsPerBlock) nBlocks++;
}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Misc functions
//////////////////////////////////////////////////////////////////////////

// Floating point compare ( no conditionals ) returns -1, 0 or 1 for < = or >
template<typename T> __GPU_CPU__ int compare(const T a, const T b, const T epsilon = T(1)/T(10000))
{
    return int((a - b) / epsilon);
}
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Bitmap handling
//////////////////////////////////////////////////////////////////////////



#pragma pack(push, 2)

struct BMPFILEHEADER
{
    unsigned short bfType;
    size_t bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    size_t bfOffBits;
};

struct BMPINFOHEADER
{
    size_t biSize;
    long biWidth;
    long biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    size_t biCompression;
    size_t biSizeImage;
    long biXPelsPerMeter;
    long biYPelsPerMeter;
    size_t biClrUsed;
    size_t biClrImportant;
};

#pragma pack(pop)

#pragma pack(push,1)

struct RGB4
{
    byte B;
    byte G;
    byte R;
    byte A;
};
//////////////////////////////////////////////////////////////////////////

struct RGB3
{
    byte B;
    byte G;
    byte R;
};
//////////////////////////////////////////////////////////////////////////

#pragma pack(pop)

//////////////////////////////////////////////////////////////////////////
// Bitmap class is a 3d array consisting of 1 or 3 planes 
//////////////////////////////////////////////////////////////////////////
template <typename T>
class TBitmap : public TCPUArray<T>
{
    ifstream m_file;

    string m_sHdrBuf;
    BMPFILEHEADER *m_pFileHdr;
    BMPINFOHEADER *m_pBmpHdr;
    int m_cbOdd;

public:
    TBitmap(pcchar pszFile, bool bGrayScale = false)
    {
        m_file.open(pszFile, ios::binary);
        if(!m_file.is_open())
            throw string("Cannot open file : ") + pszFile;

        // Load BMP file header
        size_t iFileHdrSize = sizeof(BMPFILEHEADER);
        m_sHdrBuf.resize(iFileHdrSize);
        m_file.read(&m_sHdrBuf[0], iFileHdrSize);
        m_pFileHdr = (BMPFILEHEADER*)&m_sHdrBuf[0];
        m_sHdrBuf.resize(m_pFileHdr->bfOffBits);

        m_pFileHdr = (BMPFILEHEADER*)&m_sHdrBuf[0];
        m_file.read(&m_sHdrBuf[iFileHdrSize], m_pFileHdr->bfOffBits - iFileHdrSize);

        m_pBmpHdr = (BMPINFOHEADER*)&m_sHdrBuf[iFileHdrSize];

        if(m_pBmpHdr->biBitCount != 32 && m_pBmpHdr->biBitCount != 24)
            throw string("File format can only be 32 bpp or 24 bpp");

        this->m_dim.width = m_pBmpHdr->biWidth;
        this->m_dim.height = m_pBmpHdr->biHeight;
        this->m_dim.depth = bGrayScale ? 1 : 3;
        this->resize();

        // Calculate odd bytes for stride
        dword cbPix = m_pBmpHdr->biBitCount == 24 ? 3 : 4;
        int nMod = (this->m_dim.width * cbPix) % 4;
        m_cbOdd = nMod ? 4 - nMod : 0;

        if(bGrayScale)
        {
            T3DArr<T> p(this->data());
            for(size_t y = 0; y < this->height(); ++y)
            {
                for(size_t x = 0 ; x < this->width(); ++x)
                {
                    RGB4 pix;
                    m_file.read((char*)&pix, cbPix);
                    
                    p[0][y][x] = (T)
                        ((double)pix.R * .299 + 
                        (double)pix.G * .587 + 
                        (double)pix.B * .114);
                }

                // Skip odd data
                m_file.seekg(m_cbOdd, ios_base::cur);
            }
        }
        else
        {
            T3DArr<T> p(this->data());
            for(size_t y = 0; y < this->height(); ++y)
            {
                for(size_t x = 0 ; x < this->width(); ++x)
                {
                    RGB4 pix;
                    m_file.read((char*)&pix, cbPix);
                    
                    p[0][y][x] = (T)pix.R;
                    p[1][y][x] = (T)pix.G;
                    p[2][y][x] = (T)pix.B;
                }

                // Skip odd data
                m_file.seekg(m_cbOdd, ios_base::cur);
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////

    TBitmap(int iWidth, int iHeight, bool bGrayScale = false)
    {
        size_t iFileHdrSize = sizeof(BMPFILEHEADER);
        size_t iBmpHdrSize = sizeof(BMPINFOHEADER);
        m_sHdrBuf.resize(iFileHdrSize + iBmpHdrSize);

        m_pBmpHdr = (BMPINFOHEADER*)&m_sHdrBuf[iFileHdrSize];
        m_pFileHdr = (BMPFILEHEADER*)&m_sHdrBuf[0];

        BMPFILEHEADER &bfh = *m_pFileHdr;
        BMPINFOHEADER &bih = *m_pBmpHdr;

        int nPitch = (iWidth + 3) & ~3;
        bih.biBitCount = 24;
        bih.biClrImportant = 0;
        bih.biClrUsed = 0;
        bih.biCompression = 0;
        bih.biWidth = iWidth;
        bih.biHeight = iHeight;
        bih.biPlanes = 0;
        bih.biSize = iBmpHdrSize;
        bih.biSizeImage = nPitch * iHeight * (24 / 8);
        bih.biXPelsPerMeter = 0;
        bih.biYPelsPerMeter = 0;

        bfh.bfReserved1 = 0;
        bfh.bfReserved2 = 0;
        bfh.bfSize = iFileHdrSize + iBmpHdrSize + bih.biSizeImage;
        bfh.bfType = 0x4d42;
        bfh.bfOffBits = iFileHdrSize + iBmpHdrSize;

        int nMod = (iWidth * (24 / 8)) % 4;
        m_cbOdd = nMod ? 4 - nMod : 0;

        this->m_dim.width = m_pBmpHdr->biWidth;
        this->m_dim.height = m_pBmpHdr->biHeight;
        this->m_dim.depth = bGrayScale ? 1 : 3;
        this->resize();
    }
    //////////////////////////////////////////////////////////////////////////

    void save(pcchar pszFile, bool bGrayScale = false)
    {
        ofstream ofs(pszFile, ios::binary | ios::trunc);

        // copy the bitmap header data over directly
        ofs.write(m_sHdrBuf.c_str(), m_sHdrBuf.size());

        size_t cbPix = m_pBmpHdr->biBitCount == 24 ? 3 : 4;
        char dummy[4] = {0};

        if(bGrayScale)
        {
            T3DArr<T> p(this->data());
            for(size_t y = 0; y < this->height(); ++y)
            {
                for(size_t x = 0 ; x < this->width(); ++x)
                {
                    RGB4 pix;
                    double fVal = p[0][y][x];
                    pix.R = (byte)(fVal / .299);
                    pix.G = (byte)(fVal / .587);
                    pix.B = (byte)(fVal / .114);
                    pix.A = 0;
                    ofs.write((char*)&pix, cbPix);
                }

                // Write odd bytes
                ofs.write(dummy, m_cbOdd);
            }
        }
        else
        {
            T3DArr<T> p(this->data());

            for(size_t y = 0; y < this->height(); ++y)
            {
                for(size_t x = 0 ; x < this->width(); ++x)
                {
                    RGB4 pix;
                    pix.R = (byte)p[0][y][x];
                    pix.G = (byte)p[1][y][x];
                    pix.B = (byte)p[2][y][x];
                    pix.A = 0;
                    ofs.write((char*)&pix, cbPix);
                }

                // Write odd bytes
                ofs.write(dummy, m_cbOdd);
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////

    void operator=(const TCPUArray<T> &that)
    {
        checkDims(that);
        this->m_v = that.m_v;
        this->m_pBuf = (T*)(&*this->m_v.begin());
    }
    //////////////////////////////////////////////////////////////////////////

    // Assignment from device buffer
    void operator=(const TGPUArray<T> &that)
    {
        typename TCPUArray<T>::TCopyDeviceToHost()(*this, that);
    }
    //////////////////////////////////////////////////////////////////////////    

};

#endif //__CUDALIB_H__


//////////////////////////////////////////////////////////////////////////
// XEL / BOX / PEL co-ord conversion functions
//////////////////////////////////////////////////////////////////////////

// __GPU_CPU__ void getBoxFromXel(TXY &box, TXY xel)
// {
//     box.x = xel.x + xel.y / 2;
//     box.y = xel.y;
// }
// //////////////////////////////////////////////////////////////////////////
// 
// __GPU_CPU__ void getXelFromBox(TXY &xel, TXY box)
// {
//     xel.x = box.x - box.y / 2;
//     xel.y = box.y;
// }
// //////////////////////////////////////////////////////////////////////////
// 
// // Convert XEL to PEL co ords
// __GPU_CPU__ void getPelFromXel(TFXY &pel, TFXY &xel)
// {
//     pel.x = xel.x * COS60;
//     pel.y = xel.y * SIN60;
// }
// //////////////////////////////////////////////////////////////////////////
// 
// __GPU_CPU__ void getXelFromPel(TFXY &xel, TFXY &pel)
// {
//     xel.x = pel.x / COS60;
//     xel.y = pel.y / SIN60;
// }
// //////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// Useful Co ord functions
//////////////////////////////////////////////////////////////////////////

// Get pel co-ordinate rotated by fTheta
// __GPU_CPU__ TFXY rotatePoint(TFXY ptIn, double fTheta)
// {
//     TFXY ptOut;
//     ptOut.x = ptIn.x * cos(fTheta) + ptIn.y * sin(fTheta);
//     ptOut.y = ptIn.y * cos(fTheta) - ptIn.x * sin(fTheta);
//     return ptOut;
// }
// //////////////////////////////////////////////////////////////////////////
// 
// // Get a pel/xel co-ordinate translated by x and y
// __GPU_CPU__ TFXY translate(TFXY ptIn, double x, double y)
// {
//     TFXY ptOut;
//     ptOut.x = ptIn.x + x;
//     ptOut.y = ptIn.y + y;
//     return ptOut;
// }
//////////////////////////////////////////////////////////////////////////
