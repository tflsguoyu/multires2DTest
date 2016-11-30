// #include <opencv2/core/core.hpp>
// #include <opencv2/viz/vizcore.hpp>

// int main()
// {
//     cv::viz::Viz3d window = cv::viz::Viz3d("Viz demonstration");

//     cv::Point3d min(0.25, 0.0, 0.25);
//     cv::Point3d max(0.75, 0.5, 0.75);

//     cv::viz::WCube cube(min, max, true, cv::viz::Color::red());
//     cube.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);

//     window.showWidget("Axis widget", cv::viz::WCoordinateSystem());
//     window.showWidget("Cube widget", cube);

//     while(!window.wasStopped()) {
//         window.spinOnce(1, true);
//     }

//     return 0;
// }


#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <vector>
#include <omp.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define PI 3.1415926535897932384626433832795


/*
 * Thread-safe random number generator
 */

struct RNG {
    RNG() : distrb(0.0, 1.0), engines() {}

    void init(int nworkers) {
        std::random_device rd;
        engines.resize(nworkers);
        for ( int i = 0; i < nworkers; ++i )
            engines[i].seed(rd());
    }

    double operator()() {
        int id = omp_get_thread_num();
        return distrb(engines[id]);
    }

    std::uniform_real_distribution<double> distrb;
    std::vector<std::mt19937> engines;
} rng;


/*
 * Basic data types
 */

struct Vec {
    double x, y, z;

    Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }

    Vec operator+ (const Vec &b) const  { return Vec(x+b.x, y+b.y, z+b.z); }
    Vec operator- (const Vec &b) const  { return Vec(x-b.x, y-b.y, z-b.z); }
    Vec operator* (double b) const      { return Vec(x*b, y*b, z*b); }
    bool operator== (const Vec &b) const {
        if (x == b.x && y == b.y && z == b.z) 
            return true;
        else 
            return false;
    }

    Vec mult(const Vec &b) const        { return Vec(x*b.x, y*b.y, z*b.z); }
    Vec& normalize()                    { return *this = *this * (1.0/std::sqrt(x*x+y*y+z*z)); }
    double dot(const Vec &b) const      { return x*b.x+y*b.y+z*b.z; }
    Vec cross(const Vec&b) const        { return Vec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); }
};

struct Ray {
    Vec o, d;
    Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

struct BRDF {
    virtual Vec eval(const Vec &n, const Vec &o, const Vec &i) const = 0;
    virtual void sample(const Vec &n, const Vec &o, Vec &i, double &pdf) const = 0;
    virtual bool isSpecular() const = 0;
};


/*
 * Utility functions
 */

inline double clamp(double x)   {
    return x < 0 ? 0 : x > 1 ? 1 : x;
}

inline int toInt(double x) {
    return static_cast<int>(std::pow(clamp(x), 1.0/2.2)*255+.5);
}


/*
 * Shapes
 */

struct Sphere {
    Vec p, e;           // position, emitted radiance
    double rad;         // radius
    const BRDF &brdf;   // BRDF
    
    Sphere(double rad_, Vec p_, Vec e_, const BRDF &brdf_) :
        rad(rad_), p(p_), e(e_), brdf(brdf_) {}

    double intersect(const Ray &r) const { // returns distance, 0 if nohit
        Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = op.dot(r.d), det = b*b-op.dot(op)+rad*rad;
        if ( det<0 ) return 0; else det = sqrt(det);
        return (t = b-det)>eps ? t : ((t = b+det)>eps ? t : 0);
    }
};


/*
 * Sampling functions
 */

inline void createLocalCoord(const Vec &n, Vec &u, Vec &v, Vec &w) {
    w = n;
    u = ((std::abs(w.x)>.1 ? Vec(0, 1) : Vec(1)).cross(w)).normalize();
    v = w.cross(u);
}


/*
 * BRDFs
 */

// Ideal diffuse BRDF
struct DiffuseBRDF : public BRDF {
    DiffuseBRDF(Vec kd_) : kd(kd_) {}

    Vec eval(const Vec &n, const Vec &o, const Vec &i) const {
        return kd * (1.0/PI);
    }

    void sample(const Vec &n, const Vec &o, Vec &i, double &pdf) const {
        // pdf = 2.0*PI;
        double z = sqrt(rng());
        double r = sqrt(1.0 - z * z);
        double phi = 2.0 * PI * rng();
        double x = r * cos(phi);
        double y = r * sin(phi);

        Vec u, v, w;
        createLocalCoord(n, u, v, w);
        i = (u * x + v * y + w * z).normalize();

        pdf = n.dot(i)/PI;
    }

    bool isSpecular() const {
        return false;
    }

    Vec kd;
};

Vec mirroredDirection(const Vec &_n, const Vec &_o) {
    return _n * _n.dot(_o) * 2.0 - _o;
}

// Ideal specular BRDF
struct SpecularBRDF : public BRDF {
    SpecularBRDF(Vec ks_) : ks(ks_) {}

    Vec eval(const Vec &n, const Vec &o, const Vec &i) const {
        if (i == mirroredDirection(n, o))
            return ks * (1.0/n.dot(i));
        else
            return Vec();
    }

    void sample(const Vec &n, const Vec &o, Vec &i, double &pdf) const {
        i = mirroredDirection(n,o);    
        pdf = 1.0;
    }

    bool isSpecular() const {
        return true;
    }

    Vec ks;
};


/*
 * Scene configuration
 */

// Pre-defined BRDFs
const DiffuseBRDF leftWall(Vec(.75,.25,.25)),
                  rightWall(Vec(.25,.25,.75)),
                  otherWall(Vec(.75,.75,.75)),
                  blackSurf(Vec(0.0,0.0,0.0)),
                  brightSurf(Vec(0.9,0.9,0.9));
const SpecularBRDF brightSurf_s(Vec(0.999,0.999,0.999));

// Scene: list of spheres
const Sphere spheres[] = {
    Sphere(1e5,  Vec(1e5+1,40.8,81.6),   Vec(),         leftWall),   // Left
    Sphere(1e5,  Vec(-1e5+99,40.8,81.6), Vec(),         rightWall),  // Right
    Sphere(1e5,  Vec(50,40.8, 1e5),      Vec(),         otherWall),  // Back
    Sphere(1e5,  Vec(50, 1e5, 81.6),     Vec(),         otherWall),  // Bottom
    Sphere(1e5,  Vec(50,-1e5+81.6,81.6), Vec(),         otherWall),  // Top
    Sphere(16.5, Vec(27,16.5,47),        Vec(),         brightSurf), // Ball 1
    Sphere(16.5, Vec(73,16.5,78),        Vec(),         brightSurf_s), // Ball 2
    Sphere(5.0,  Vec(50,70.0,81.6),      Vec(50,50,50), blackSurf)   // Light
};

// Camera position & direction
const Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).normalize());


/*
 * Global functions
 */

bool intersect(const Ray &r, double &t, int &id) {
    double n = sizeof(spheres)/sizeof(Sphere), d, inf = t = 1e20;
    for ( int i = int(n); i--;) if ( (d = spheres[i].intersect(r))&&d<t ) { t = d; id = i; }
    return t<inf;
}


/*
 * KEY FUNCTION: radiance estimator
 */

// Vec emittedRadiance(const Sphere &);
// Vec indirectedRadiance(const Vec &, const Vec &, const Sphere&, int, bool);
// Vec directedRadiance(const Vec &, const Vec &, const Sphere&, bool);
Vec reflectedRadiance(const Vec &, const Vec &, const Sphere&, int, bool);
Vec Radiance(const Vec &, const Vec &, const Sphere &, int, bool);
// Vec receivedRadiance(const Ray &, int, bool);

Vec emittedRadiance(const Sphere &obj) {

    return obj.e;
}

void luminaireSample(const Sphere& obj, Vec &p, Vec &n, double &pdf) {

    double sigma1 = rng();
    double sigma2 = rng();
    double z = 2*sigma1-1;
    double x = sqrt(1-z*z)*cos(2*PI*sigma2);
    double y = sqrt(1-z*z)*sin(2*PI*sigma2);

    n = Vec(x,y,z);
    p = obj.p + n*obj.rad;
    pdf = 1/(4*PI*obj.rad*obj.rad);
}

Vec directedRadiance(const Vec &x, const Vec &o, const Sphere& obj, bool flag) {

    const BRDF &brdf = obj.brdf;
    
    if (brdf.isSpecular())
        return Vec();
    else
    {
        // x info
        Vec n = (x - obj.p).normalize();            // The normal direction
        if ( n.dot(o) < 0 ) n = n*-1.0;

        const BRDF &brdf = obj.brdf;                // Surface BRDF at x

        // light info
        const Sphere &obj_l = spheres[7];           // light source 
        Vec x_l, n_l;
        double pdf_l;
        luminaireSample(obj_l, x_l, n_l, pdf_l);

        Vec i = (x_l - x).normalize();
        double disXtoXl = (x_l-x).dot(x_l-x);

        // visibility
        double t;
        int id;
        intersect(Ray(x,i), t, id);
        Vec x_hit = x + i*t;                        // The intersection point

        if ( (x_hit-x_l).dot(x_hit-x_l) < 0.0001 )
            return emittedRadiance(obj_l).mult(brdf.eval(n,o,i)) * n.dot(i) * n_l.dot(Vec()-i) * (1/(disXtoXl*pdf_l));
        else
            return Vec();
    }
}

Vec indirectedRadiance(const Vec &x, const Vec &o, const Sphere& obj, int depth, bool flag) {

    double p;

    if (depth <= 5) 
        p = 1.0;
    else
        p = 0.9;
    
    if (rng() > p) return Vec(); 

    Vec n = (x - obj.p).normalize();            // The normal direction
    if ( n.dot(o) < 0 ) n = n*-1.0;

    const BRDF &brdf = obj.brdf;                // Surface BRDF at x
    Vec i;
    double pdf;
    brdf.sample(n, o, i, pdf);

    // compute intersection point, outgoing direction
    double t;                                   // Distance to intersection
    int id = 0;                                 // id of intersected sphere

    if ( !intersect(Ray(x,i), t, id) ) {return Vec(); }  // if miss, return black
    const Sphere &obj_y = spheres[id];            // the hit object

    Vec y = x + i*t;                        // The intersection point
    Vec o_y = (Vec() - i).normalize();          // The outgoing direction (= -r.d)

    
    if (brdf.isSpecular())
        return Radiance(y, o_y, obj_y, depth+1, flag).mult(brdf.eval(n, o, i)) * n.dot(i) * (1/(pdf*p));
    else
        return reflectedRadiance(y, o_y, obj_y, depth+1, flag).mult(brdf.eval(n, o, i)) * n.dot(i) * (1/(pdf*p));


}

Vec reflectedRadiance(const Vec &x, const Vec &o, const Sphere& obj, int depth, bool flag) {

    return directedRadiance(x, o, obj, flag) + indirectedRadiance(x, o, obj, depth, flag);
    // return directedRadiance(x, o, obj, flag);

}

Vec Radiance(const Vec &x, const Vec &o, const Sphere &obj, int depth, bool flag) {
    
    return emittedRadiance(obj) + reflectedRadiance(x,o,obj,depth,flag);
}


Vec receivedRadiance(const Ray &r, int depth, bool flag) {

    // compute intersection point, outgoing direction
    double t;                                   // Distance to intersection
    int id = 0;                                 // id of intersected sphere

    if ( !intersect(r, t, id) ) {return Vec(); }  // if miss, return black
    const Sphere &obj = spheres[id];            // the hit object

    Vec x = r.o + r.d*t;                        // The intersection point
    Vec o = (Vec() - r.d).normalize();          // The outgoing direction (= -r.d)
    
    return Radiance(x,o,obj,depth,flag);
}




/*
 * Main function (do not modify)
 */

int main(int argc, char *argv[]) {
    int nworkers = omp_get_num_procs();
    omp_set_num_threads(nworkers);
    rng.init(nworkers);

    int w = 640, h = 480, samps = argc==2 ? atoi(argv[1])/4 : 1; // # samples
    Vec cx = Vec(w*.5135/h), cy = (cx.cross(cam.d)).normalize()*.5135;    
    std::vector<Vec> c(w*h);

#pragma omp parallel for schedule(dynamic, 1)
    for ( int y = 0; y < h; y++ ) {
        for ( int x = 0; x < w; x++ ) {
            const int i = (h - y - 1)*w + x;

            for ( int sy = 0; sy < 2; ++sy ) {
                for ( int sx = 0; sx < 2; ++sx ) {
                    Vec r;
                    for ( int s = 0; s<samps; s++ ) {
                        double r1 = 2*rng(), dx = r1<1 ? sqrt(r1)-1 : 1-sqrt(2-r1);
                        double r2 = 2*rng(), dy = r2<1 ? sqrt(r2)-1 : 1-sqrt(2-r2);
                        Vec d = cx*(((sx+.5 + dx)/2 + x)/w - .5) +
                            cy*(((sy+.5 + dy)/2 + y)/h - .5) + cam.d;
                        r = r + receivedRadiance(Ray(cam.o, d.normalize()), 1, true)*(1./samps);
                    }
                    c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
                }
            }
        }
#pragma omp critical
        fprintf(stderr,"\rRendering (%d spp) %6.2f%%",samps*4,100.*y/(h-1));
    }
    fprintf(stderr, "\n");

    // Write resulting image to a PPM file
    // FILE *f = fopen("image.ppm", "w");
    // fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
    // for ( int i = 0; i<w*h; i++ )
    //     fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
    // fclose(f);

    printf("%f,%f,%f\n",c[0].x,c[0].y,c[0].z);
    printf("1 OK\n");

    // Write resulting image to a EXR file using OpenCV
    cv::Mat image(h,w,CV_32FC3);
    printf("2 OK\n");
	for ( int y = 0; y < h; y++ ) {
		for ( int x = 0; x < w; x++ ) {
			const int i = y * w + x;
			// printf("%d %d %d\n",y,x,i);
			image.at<cv::Vec3f>(cv::Point(x,y))[0] = c[i].z;
			image.at<cv::Vec3f>(cv::Point(x,y))[1] = c[i].y;
			image.at<cv::Vec3f>(cv::Point(x,y))[2] = c[i].x;
		}
	}
	// cv::cvtColor(image, image, CV_BGR2RGB);
	cv::imwrite("image.exr",image);
	// cv::Mat img = cv::imread("image.exr");
	// cv::imshow("rendering",img);
	// cv::waitKey(0);

    return 0;
}