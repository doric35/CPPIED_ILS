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
inline cv::Mat draw_final_frame(const std::vector<segment>& path,
                                const Eigen::Ref<const Eigen::VectorXd>& coverage,
                                T& helper,
                                int height, int width,
                                int time, double scale){
    double min_req = 1 - std::exp(-helper.problem.req.minCoeff());
    double spread = 1.0 - min_req;
    cv::Mat coverage_img(helper.geometry.n_rows, helper.geometry.n_cols, CV_8UC1);
    for (int y = 0; y < helper.geometry.n_rows; ++y) {
        for (int x = 0; x < helper.geometry.n_cols; ++x) {
            double norm = std::clamp(
                (coverage((y * helper.geometry.n_cols) + x) - min_req) / spread,
                0.0, 1.0);
            coverage_img.at<uchar>(y,x) = static_cast<uchar>(255.0 * norm);
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
    draw_path(path, helper, frame, time, scale);
    return frame;
}

inline void draw_colorbar(const std::string& file_name,
                          double min_val, double max_val,
                          int height, int width) {
    const int bar_width = width / 10;
    const int padding = 15;
    const int img_height = height + 2 * padding;

    // Measure label text once to compute the tightest possible right margin
    constexpr double font_scale = 0.8;
    constexpr int    font_face  = cv::FONT_HERSHEY_COMPLEX;
    int baseline = 0;
    cv::Size label_size = cv::getTextSize("0.00", font_face, font_scale, 1, &baseline);
    const int img_width = bar_width + 6 + 9 + label_size.width + 3;  // bar + tick + gap + text + buffer

    // Vertical gradient: top = max (255 = red in JET), bottom = min (0 = blue)
    cv::Mat bar(height, bar_width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        auto val = static_cast<uchar>(255.0 * (1.0 - double(y) / double(height - 1)));
        for (int x = 0; x < bar_width; ++x)
            bar.at<uchar>(y, x) = val;
    }
    cv::Mat colored;
    cv::applyColorMap(bar, colored, cv::COLORMAP_JET);

    // Place bar with padding at top and bottom so labels are never clipped
    cv::Mat img(img_height, img_width, CV_8UC3, cv::Scalar(255, 255, 255));
    colored.copyTo(img(cv::Rect(0, padding, bar_width, height)));

    // Tick marks and labels at evenly spaced values
    constexpr int n_ticks = 5;
    for (int i = 0; i <= n_ticks; ++i) {
        double t = double(i) / double(n_ticks);
        double val = min_val + t * (max_val - min_val);
        int y = padding + height - 1 - static_cast<int>(std::round(t * double(height - 1)));

        cv::line(img, {bar_width, y}, {bar_width + 6, y}, cv::Scalar(0, 0, 0), 1);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << val;
        cv::putText(img, oss.str(),
                    {bar_width + 9, y + label_size.height / 2},
                    font_face, font_scale,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    cv::imwrite(file_name, img);
}

template <typename T>
inline void write_visulization(
        const std::string& file_name,
        const std::string& img_file,
        const std::string& sol_frame_fn,
        cppied_solution& s,
        T& helper,
        int max_time){
    double scale = 512.0 / double(helper.geometry.n_cols);
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
            cv::VideoWriter::fourcc('a','v','c','1'),
            fps,                      // FPS
            cv::Size(width, height)  // frame size
    );

    if (!writer.isOpened()) {
        std::cerr << "[ERROR] VideoWriter failed to open: " << file_name
                  << "\n  Check that the mp4v codec is available in this OpenCV build."
                  << std::endl;
        return;
    }

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
    frame = draw_final_frame(sol_cp.path, exp_coverage, helper,
                             height, width, s.path.size()-1, scale);
    cv::imwrite(sol_frame_fn, frame);
    std::string png_to_pdf = "magick ";
    std::string inst_name = sol_frame_fn.substr(
            0,
            sol_frame_fn.find('.')
    );
    {
        double min_req = 1.0 - std::exp(-helper.problem.req.minCoeff());
        std::string colorbar_fn = inst_name + "_colorbar.png";
        draw_colorbar(colorbar_fn, min_req, 1.0, height, 200);
        std::string cb_pdf = "magick " + colorbar_fn + " " + inst_name + "_colorbar.pdf";
        int cb_flag = std::system(cb_pdf.c_str());
        if (cb_flag)
            std::cerr << "[WARNING] Non 0 return when writing colorbar png to pdf : flag " << cb_flag << std::endl;
    }
    png_to_pdf += sol_frame_fn + " " + inst_name + ".pdf";
    int r_flag = std::system(png_to_pdf.c_str());
    if (r_flag){
        std::cerr << "[WARNING] Non 0 return when writing image png to pdf : flag " << r_flag << std::endl;
    }

    frame = seabed_to_frame(helper.problem.seabed,
                            height, width);
    cv::imwrite(img_file, frame);
    inst_name = img_file.substr(
            0,
            img_file.find('.')
            );
    png_to_pdf = "magick " + img_file + " " + inst_name + ".pdf";
    r_flag = std::system(png_to_pdf.c_str());
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
