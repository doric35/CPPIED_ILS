#pragma once

#include "utils.hpp"

//Handle meta-data
struct cppied_context {
    std::unordered_map<std::string, std::string> config;
    std::vector<std::string> csv_log;
    int max_time;
    std::chrono::high_resolution_clock::time_point start_time, end_time;

    cppied_context(std::unordered_map<std::string, std::string> p_config)
            : config(std::move(p_config)), max_time(0)
    {
        max_time = std::stoi(config["TIME"]);
    }
};

struct cost_t {
    int length;
    int turns;

    cost_t& operator+=(const cost_t& other) {
        length += other.length;
        turns += other.turns;
        return *this;
    }

    friend cost_t operator+(cost_t a, const cost_t& b) {
        a += b;
        return a;
    }

    cost_t& operator-=(const cost_t& other) {
        length -= other.length;
        turns -= other.turns;
        return *this;
    }

    cost_t operator*(int k) const {
        return {length * k, turns * k};
    }

    cost_t operator+(int k) const {
        return {length + k, turns + k};
    }

    cost_t operator-(int k) const {
        return {length - k, turns - k};
    }

    friend cost_t operator-(cost_t a, const cost_t& b) {
        a -= b;
        return a;
    }

    auto operator<=>(const cost_t&) const = default;
};

struct Point2F{
    float x,y;
    static float L1(const Point2F& a, const Point2F& b){
        return std::abs(a.x - b.x) + std::abs(a.y - b.y);
    }
    auto operator<=>(const Point2F&) const = default;
};

inline std::ostream& operator<<(std::ostream& os, const cost_t& c) {
    return os << "{l=" << c.length << ", t=" << c.turns << "}";
}

struct interval{
    cost_t gain;
    int x1;
    int x2;
    int y;
    bool rotated;
    int id;
};

struct iCoordinate{
    int x;
    int y;
};

struct iRectangle{
    iCoordinate ul;
    iCoordinate lr;
};

//Handle solutions
template <typename PathContainer>
struct cppied_solution_base {
    PathContainer path;
    Eigen::VectorXd coverage;
    cost_t cost{0,0};

    using path_type = PathContainer;
};

enum class direction : uint8_t { E = 0, W = 1, S = 2, N = 3 };

struct segment {
    int source;
    int target;
    auto operator<=>(const segment&) const = default;
};

inline std::ostream& operator<<(std::ostream& os, const segment& s) {
    return os << "{src=" << s.source << ", tgt=" << s.target << "}";
}

using cppied_solution =
        cppied_solution_base<std::vector<segment>>;

inline std::ostream& operator<<(std::ostream& os, const cppied_solution& s) {
    os << "COST_SECTION\n";
    os << s.cost << "\n";
    os << "SEGMENT_SECTION\n";
    for (auto& segment : s.path)
        os << segment << "\n";
    os << "COVERAGE_SECTION\n";
    for (int i=0; i<s.coverage.rows(); ++i)
        os << s.coverage.row(i) << "\n";
    os << "EOF" << std::endl;
    return os;
}

inline cv::Point map_to_image(Point2F& p,
                              double scale){
    int px = static_cast<int>(p.x * scale);
    int py = static_cast<int>(p.y * scale);
    return {px, py};
}

inline cv::Mat seabed_to_frame(
        const Eigen::Ref<const MatrixXdRow<int>>& seabed,
        int height, int width){
    cv::Mat seabed_img(int(seabed.rows()),
                       int(seabed.cols()), CV_8UC1);

    int min_val = seabed.minCoeff();
    int max_val = seabed.maxCoeff() + 1;

    for (int y = 0; y < seabed.rows(); ++y) {
        for (int x = 0; x < seabed.cols(); ++x) {
            double norm = double(seabed(y, x) - min_val) / double (max_val - min_val + 1e-9);
            seabed_img.at<uchar>(y,x) = static_cast<uchar>(255 * (1.0 - norm));
        }
    }
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(255,255,255));

    cv::Mat resized;
    cv::resize(seabed_img, resized, cv::Size(width, height));
    cv::Mat colored;
    cv::applyColorMap(resized, colored, cv::COLORMAP_OCEAN);

    colored.copyTo(frame(cv::Rect(0,0,width, height)));
    return frame;
}

template<typename T>
inline cv::Mat coverage_to_frame(const Eigen::Ref<const Eigen::VectorXd>& coverage,
                                 T& helper,
                                 int height, int width){
    cv::Mat coverage_img(helper.geometry.n_rows, helper.geometry.n_cols, CV_8UC1);

    for (int y = 0; y < helper.geometry.n_rows; ++y) {
        for (int x = 0; x < helper.geometry.n_cols; ++x) {
            coverage_img.at<uchar>(y,x) = static_cast<uchar>(
                    255 * coverage((y * helper.geometry.n_cols) + x)
                    );
        }
    }

    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(255,255,255));

    cv::Mat resized;
    cv::Mat colored;
    cv::resize(coverage_img, resized,
               cv::Size(width, height),
               0, 0,
               cv::INTER_NEAREST);
    cv::applyColorMap(resized, colored, cv::COLORMAP_JET);

    colored.copyTo(frame(cv::Rect(0,0,width, height)));
    return frame;
}

template <typename T>
inline void draw_path(const std::vector<segment>& path,
                      T& helper,
                      cv::Mat& frame, int time, double scale){
    cv::Point first = map_to_image(helper.problem.vertex[path[0].source], scale);
    cv::Point second;
    if (helper.geometry.is_node(path[0].target))
        second = map_to_image(helper.problem.vertex[path[0].target], scale);
    else
        second = map_to_image(helper.problem.vertex[path[0].source], scale);
    cv::line(frame,
             first,
             second,
             cv::Scalar(0,0,0),
             20);
    for (int t=1; t <= time; t++){
        first = map_to_image(helper.problem.vertex[path[t].source], scale);
        cv::line(frame,
                 second,
                 first,
                 cv::Scalar(0,0,0),
                 20);
        if (helper.geometry.is_node(path[t].target))
            second = map_to_image(helper.problem.vertex[path[t].target], scale);
        else
            second = map_to_image(helper.problem.vertex[path[t].source], scale);

        cv::line(frame,
                 first,
                 second,
                 cv::Scalar(0,0,0),
                 20);
    }
    if (helper.geometry.is_node(path[time].target)) {
        cv::circle(frame,
                   map_to_image(helper.problem.vertex[path[time].target], scale),
                   20, cv::Scalar(0, 0, 0));
    } else{
        cv::circle(frame,
                   map_to_image(helper.problem.vertex[path[time].source], scale),
                   20, cv::Scalar(0, 0, 0));
    }
}

template <typename T>
inline void write_visulization(
        const std::string& file_name,
        const std::string& img_file,
        cppied_solution& s,
        T& helper,
        int max_time){
    double scale = 600.0 / double(helper.geometry.n_cols);
    int height, width;
    height = static_cast<int>(std::ceil(scale * double(helper.geometry.n_rows)));
    width = static_cast<int>(std::ceil(scale * double(helper.geometry.n_cols)));


    cppied_solution sol_cp = s;

    int fps;
    fps = std::max(1,
                   static_cast<int>(std::ceil(double(s.path.size()) / double(max_time)))
                   );

    cv::VideoWriter writer(
            file_name,
            cv::VideoWriter::fourcc('m','p','4','v'),
            fps,                      // FPS
            cv::Size(width, height)  // frame size
    );

    sol_cp.coverage = Eigen::VectorXd::Zero(sol_cp.coverage.size());
    Eigen::VectorXd exp_coverage(sol_cp.coverage.size());
    exp_coverage.setZero();

    exp_coverage.array() = 1.0 - (-sol_cp.coverage.array()).exp();
    cv::Mat frame = coverage_to_frame(exp_coverage, helper, height, width);
    writer.write(frame);
    for (int i = 0; i<s.path.size(); ++i){
        helper.coverage.insert(sol_cp, sol_cp.path[i]);
        exp_coverage.array() = 1.0 - (-sol_cp.coverage.array()).exp();
        frame = coverage_to_frame(exp_coverage, helper, height, width);
        draw_path(sol_cp.path, helper, frame, i, scale);
        writer.write(frame);
    }
    writer.release();
    frame = seabed_to_frame(helper.problem.seabed,
                            height, width);
    cv::imwrite(img_file, frame);
    std::string png_to_pdf = "magick ";
    std::string inst_name = img_file.substr(
            0,
            img_file.find('.')
            );
    png_to_pdf += img_file + " " + inst_name + ".pdf";
    int r_flag = std::system(png_to_pdf.c_str());
    if (r_flag){
        std::cerr << "[WARNING] Non 0 return when writing image png to pdf : flag " << r_flag << std::endl;
    }
}

using cppied_solution_list =
        cppied_solution_base<std::list<segment>>;

inline direction flip_direction(direction d) {
    switch (d) {
        case direction::E: return direction::W;
        case direction::W: return direction::E;
        case direction::S: return direction::N;
        case direction::N: return direction::S;
    }
    __builtin_unreachable();
}

enum class pattern : uint8_t {
    Same,
    Opposite,
    Orthogonal
};

inline pattern classify(direction a, direction b) {
    if (a == b) return pattern::Same;
    if (a == flip_direction(b)) return pattern::Opposite;
    return pattern::Orthogonal;
}

constexpr int direction_idx(direction d) {
    return static_cast<int>(d);
}
