#pragma once
#include "env.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

// Shared topology-aware features for TicTacToe (3-in-a-row) and Connect Four (4-in-a-row).
inline bool board_structural_primitive(const std::string& name, const std::vector<float>& state,
                                       int rows, int cols, int win_length, float& value) {
  const int cells = rows * cols;
  if ((int)state.size() != 2 * cells) return false;
  int own_open = 0, opp_open = 0;
  std::set<int> own_wins, opp_wins;
  int line_count = 0;
  for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
    const int drs[] = {0, 1, 1, 1}, dcs[] = {1, 0, 1, -1};
    for (int direction = 0; direction < 4; direction++) {
      int er = r + (win_length - 1) * drs[direction];
      int ec = c + (win_length - 1) * dcs[direction];
      if (er < 0 || er >= rows || ec < 0 || ec >= cols) continue;
      line_count++;
      int own_count = 0, opp_count = 0, empty_count = 0, empty_index = -1;
      for (int k = 0; k < win_length; k++) {
        int rr = r + k * drs[direction], cc = c + k * dcs[direction], idx = rr * cols + cc;
        bool own = state[idx] > 0.5f, opp = state[cells + idx] > 0.5f;
        own_count += own; opp_count += opp;
        if (!own && !opp) { empty_count++; empty_index = idx; }
      }
      if (own_count > 0 && opp_count == 0) own_open++;
      if (opp_count > 0 && own_count == 0) opp_open++;
      if (own_count == win_length - 1 && opp_count == 0 && empty_count == 1) own_wins.insert(empty_index);
      if (opp_count == win_length - 1 && own_count == 0 && empty_count == 1) opp_wins.insert(empty_index);
    }
  }
  if (name == "board_own_open_lines") value = (float)own_open / std::max(1, line_count);
  else if (name == "board_opponent_open_lines") value = (float)opp_open / std::max(1, line_count);
  else if (name == "board_own_immediate_wins") value = (float)own_wins.size() / cells;
  else if (name == "board_opponent_immediate_wins") value = (float)opp_wins.size() / cells;
  else if (name == "board_center_control") {
    float own = 0.f, opp = 0.f;
    if (cols == 3) { own = state[(rows / 2) * cols + 1]; opp = state[cells + (rows / 2) * cols + 1]; }
    else for (int r = 0; r < rows; r++) for (int c = 2; c <= 4; c++) {
      int idx = r * cols + c; own += state[idx]; opp += state[cells + idx];
    }
    value = cols == 3 ? own - opp : (own - opp) / (rows * 3.f);
  } else if (name == "board_occupancy") {
    int occupied = 0;
    for (int i = 0; i < cells; i++) occupied += state[i] > 0.5f || state[cells + i] > 0.5f;
    value = (float)occupied / cells;
  } else return false;
  return true;
}

// FIXED mode test env: 4 state feats, 2 actions.
struct CartPole : Env {
  float x, xd, th, thd; int t = 0;
  bool per_action() const override { return false; }
  int n_state() const override { return 4; }
  int n_actions() const override { return 2; }
  void reset(Rng& r) override { x = r.range(-.05f, .05f); xd = r.range(-.05f, .05f); th = r.range(-.05f, .05f); thd = r.range(-.05f, .05f); t = 0; }
  int legal(int* o) override { o[0] = 0; o[1] = 1; return 2; }
  void state_feat(float* o) override { o[0] = x / 2.4f; o[1] = xd / 2.f; o[2] = th / 0.21f; o[3] = thd / 2.f; }
  bool additional_primitive(const std::string& name, const std::vector<float>& s, float& value) const override {
    if (name != "state_boundary_margin" || s.size() < 3) return false;
    value = 1.f - std::max(std::fabs(s[0]), std::fabs(s[2]));
    return true;
  }
  float step(int a, bool& done, Rng&) override {
    const float g = 9.8f, mc = 1.f, mp = .1f, mt = mc + mp, l = .5f, pml = mp * l, tau = .02f;
    float f = a ? 10.f : -10.f, c = std::cos(th), s = std::sin(th);
    float tmp = (f + pml * thd * thd * s) / mt;
    float tha = (g * s - c * tmp) / (l * (4.f / 3.f - mp * c * c / mt)), xa = tmp - pml * tha * c / mt;
    x += tau * xd; xd += tau * xa; th += tau * thd; thd += tau * tha; t++;
    done = std::fabs(x) > 2.4f || std::fabs(th) > 12.f * 3.14159265f / 180.f || t >= 500;
    return 1.f;
  }
  std::unique_ptr<Env> make() const override { return std::make_unique<CartPole>(); }
};

// PER_ACTION mode test env: tic-tac-toe vs random opponent. 18 state feats; nf action feats ("concepts"):
// 0 wins_now, 1 blocks_opp, 2 is_center, 3 is_corner. nf can be raised mid-run to test add_input.
struct TicTacToe : Env {
  int8_t b[9]; int nf = 4; int feature_mode = 0; // 0=handcrafted, 1=deterministic random control
  int opp = 0;   // opponent policy: 0 = uniformly random, 1 = basic strategic (win if possible, else block, else random)
  bool per_action() const override { return true; }
  int n_state() const override { return 18; }
  int n_action_feat() const override { return nf; }
  int n_actions() const override { return 9; }
  int n_action_tactics() const override { return 2; }
  void action_tactics(int a, float* out) override {
    int8_t t[9]; for (int i = 0; i < 9; i++) t[i] = b[i];
    t[a] = 1; out[0] = win(t, 1) ? 1.f : 0.f;
    t[a] = 2; out[1] = win(t, 2) ? 1.f : 0.f;
  }
  static bool win(const int8_t* b, int p) {
    static const int Ls[8][3] = {{0,1,2},{3,4,5},{6,7,8},{0,3,6},{1,4,7},{2,5,8},{0,4,8},{2,4,6}};
    for (auto& l : Ls) if (b[l[0]] == p && b[l[1]] == p && b[l[2]] == p) return true;
    return false;
  }
  bool full() const { for (int i = 0; i < 9; i++) if (!b[i]) return false; return true; }
  void opp_move(Rng& r) {
    int m[9], n = 0; for (int i = 0; i < 9; i++) if (!b[i]) m[n++] = i;
    if (!n) return;
    if (opp == 1) {
      for (int pass = 0; pass < 2; pass++) {            // pass 0: win (as 2); pass 1: block (player 1)
        int who = pass == 0 ? 2 : 1;
        for (int k = 0; k < n; k++) { b[m[k]] = who; bool w = win(b, who); b[m[k]] = 0; if (w) { b[m[k]] = 2; return; } }
      }
    }
    b[m[r.below(n)]] = 2;
  }
  void reset(Rng& r) override { for (auto& c : b) c = 0; if (r.chance(.5f)) opp_move(r); }
  int legal(int* o) override { int n = 0; for (int i = 0; i < 9; i++) if (!b[i]) o[n++] = i; return n; }
  void state_feat(float* o) override { for (int i = 0; i < 9; i++) { o[i] = b[i] == 1; o[9 + i] = b[i] == 2; } }
  bool additional_primitive(const std::string& name, const std::vector<float>& state, float& value) const override {
    return board_structural_primitive(name, state, 3, 3, 3, value);
  }
  void action_feat(int a, float* o) override {
    int8_t t[9]; std::copy(b, b + 9, t);
    float f[4];
    if (feature_mode == 1) {
      // Deterministic pseudo-random features: same board/action gives same values,
      // but they carry no intentionally useful game semantics.
      uint32_t h = 2166136261u ^ (uint32_t)a;
      for (int i = 0; i < 9; i++) { h ^= (uint32_t)(b[i] + 3); h *= 16777619u; }
      for (int i = 0; i < 4; i++) { h ^= (h >> 13); h *= 0x5bd1e995u; f[i] = ((h & 0xffffu) / 32767.5f) - 1.f; }
    } else {
      t[a] = 1; f[0] = win(t, 1); t[a] = 2; f[1] = win(t, 2);
      f[2] = a == 4; f[3] = (a == 0 || a == 2 || a == 6 || a == 8);
    }
    for (int i = 0; i < nf; i++) o[i] = f[i];
  }
  float step(int a, bool& done, Rng& r) override {
    b[a] = 1;
    if (win(b, 1)) { done = true; return 1.f; }
    if (full()) { done = true; return 0.f; }
    opp_move(r);
    if (win(b, 2)) { done = true; return -1.f; }
    if (full()) { done = true; return 0.f; }
    return 0.f;
  }
  std::unique_ptr<Env> make() const override { auto e = std::make_unique<TicTacToe>(); e->nf = nf; e->feature_mode = feature_mode; e->opp = opp; return e; }
};

struct ConnectFour : Env {
  static constexpr int kRows = 6;
  static constexpr int kCols = 7;
  int8_t board[kRows][kCols]{};
  int opp = 0;   // 0 = uniformly random opponent, 1 = basic strategic (win in one, else block in one, else random)

  bool per_action() const override { return true; }
  int n_state() const override { return 2 * kRows * kCols; }
  int n_actions() const override { return kCols; }
  void reset(Rng& r) override {
    for (auto& row : board) for (auto& cell : row) cell = 0;
    if (r.chance(.5f)) {
      int legal_cols[kCols];
      int n = legal(legal_cols);
      drop(legal_cols[r.below(n)], 2);
    }
  }
  int legal(int* out) override {
    int n = 0;
    for (int col = 0; col < kCols; col++) if (board[0][col] == 0) out[n++] = col;
    return n;
  }
  void state_feat(float* out) override {
    for (int row = 0; row < kRows; row++) {
      for (int col = 0; col < kCols; col++) {
        int i = row * kCols + col;
        out[i] = board[row][col] == 1;
        out[kRows * kCols + i] = board[row][col] == 2;
      }
    }
  }
  bool additional_primitive(const std::string& name, const std::vector<float>& state, float& value) const override {
    return board_structural_primitive(name, state, kRows, kCols, 4, value);
  }
  int n_action_tactics() const override { return 2; }
  void action_tactics(int col, float* out) override {
    out[0] = out[1] = 0.f;
    for (int who = 1; who <= 2; who++) {
      int8_t t[kRows][kCols]; std::copy(&board[0][0], &board[0][0] + kRows * kCols, &t[0][0]);
      for (int row = kRows - 1; row >= 0; row--) if (t[row][col] == 0) { t[row][col] = (int8_t)who; break; }
      if (wins(t, who)) out[who - 1] = 1.f;
    }
  }
  static bool wins(const int8_t cells[kRows][kCols], int player) {
    static const int dr[] = {0, 1, 1, 1};
    static const int dc[] = {1, 0, 1, -1};
    for (int row = 0; row < kRows; row++) for (int col = 0; col < kCols; col++) {
      if (cells[row][col] != player) continue;
      for (int d = 0; d < 4; d++) {
        int end_row = row + 3 * dr[d], end_col = col + 3 * dc[d];
        if (end_row < 0 || end_row >= kRows || end_col < 0 || end_col >= kCols) continue;
        bool four = true;
        for (int step = 1; step < 4; step++)
          if (cells[row + step * dr[d]][col + step * dc[d]] != player) four = false;
        if (four) return true;
      }
    }
    return false;
  }
  float step(int action, bool& done, Rng& r) override {
    if (action < 0 || action >= kCols || board[0][action] != 0)
      throw std::invalid_argument("illegal Connect Four column");
    drop(action, 1);
    if (wins(board, 1)) { done = true; return 1.f; }
    int legal_cols[kCols];
    int n = legal(legal_cols);
    if (!n) { done = true; return 0.f; }
    int opponent_action = legal_cols[r.below(n)];
    if (opp == 1) {
      int pick = -1;
      for (int pass = 0; pass < 2 && pick < 0; pass++) {
        int who = pass == 0 ? 2 : 1;
        for (int k = 0; k < n && pick < 0; k++) {
          int8_t t[kRows][kCols]; std::copy(&board[0][0], &board[0][0] + kRows * kCols, &t[0][0]);
          for (int row = kRows - 1; row >= 0; row--) if (t[row][legal_cols[k]] == 0) { t[row][legal_cols[k]] = (int8_t)who; break; }
          if (wins(t, who)) pick = legal_cols[k];
        }
      }
      if (pick >= 0) opponent_action = pick;
    }
    drop(opponent_action, 2);
    if (wins(board, 2)) { done = true; return -1.f; }
    done = legal(legal_cols) == 0;
    return 0.f;
  }
  std::unique_ptr<Env> make() const override { return [&] { auto e = std::make_unique<ConnectFour>(); e->opp = opp; return e; }(); }

private:
  void drop(int col, int player) {
    for (int row = kRows - 1; row >= 0; row--) {
      if (board[row][col] == 0) { board[row][col] = (int8_t)player; return; }
    }
    throw std::invalid_argument("Connect Four column is full");
  }
};

struct Nim : Env {
  static constexpr int kMaxTake = 3;
  static constexpr int kMaxPile = 21;
  int remaining = 0;
  int opp = 0;   // 0 = uniformly random opponent, 1 = optimal opponent (leaves a multiple of 4 whenever possible)

  bool per_action() const override { return true; }
  int n_state() const override { return 1; }
  int n_actions() const override { return kMaxTake; }
  void reset(Rng& r) override { remaining = 10 + r.below(kMaxPile - 9); }
  int legal(int* out) override {
    int n = 0;
    for (int take = 1; take <= kMaxTake && take <= remaining; take++) out[n++] = take - 1;
    return n;
  }
  void state_feat(float* out) override { out[0] = (float)remaining / kMaxPile; }
  float step(int action, bool& done, Rng& r) override {
    int take = action + 1;
    if (take < 1 || take > kMaxTake || take > remaining)
      throw std::invalid_argument("illegal Nim take");
    remaining -= take;
    if (remaining == 0) { done = true; return 1.f; }
    int opponent_take = 1 + r.below(std::min(kMaxTake, remaining));
    if (opp == 1 && remaining % 4 != 0) opponent_take = remaining % 4;
    remaining -= opponent_take;
    if (remaining == 0) { done = true; return -1.f; }
    done = false;
    return 0.f;
  }
  std::unique_ptr<Env> make() const override { return [&] { auto e = std::make_unique<Nim>(); e->opp = opp; return e; }(); }
};

struct LineWorld : Env {
  static constexpr int kGoal = 10;
  static constexpr int kMaxSteps = 30;
  int position = 0;
  int steps = 0;

  bool per_action() const override { return false; }
  int n_state() const override { return 2; }
  int n_actions() const override { return 2; }
  void reset(Rng& r) override { position = r.below(5); steps = 0; }
  int legal(int* out) override {
    int n = 0;
    if (position > 0) out[n++] = 0;
    if (position < kGoal) out[n++] = 1;
    return n;
  }
  void state_feat(float* out) override {
    out[0] = (float)position / kGoal;
    out[1] = 1.f - (float)steps / kMaxSteps;
  }
  float step(int action, bool& done, Rng&) override {
    if ((action != 0 && action != 1) || (action == 0 && position == 0) || (action == 1 && position == kGoal))
      throw std::invalid_argument("illegal LineWorld action");
    position += action == 1 ? 1 : -1;
    steps++;
    done = position == kGoal || steps >= kMaxSteps;
    if (position == kGoal) return 1.f;
    if (done) return -0.1f;
    return -0.01f;
  }
  std::unique_ptr<Env> make() const override { return std::make_unique<LineWorld>(); }
};

struct GridWorld : Env {
  int width, height, max_turns;
  int start_x, start_y, goal_x, goal_y;
  int x, y, turns = 0;
  std::vector<std::pair<int, int>> walls, hazards;

  GridWorld(int w, int h, int max_t, int sx, int sy, int gx, int gy,
            std::vector<std::pair<int, int>> ws,
            std::vector<std::pair<int, int>> hs)
      : width(w), height(h), max_turns(max_t), start_x(sx), start_y(sy),
        goal_x(gx), goal_y(gy), x(sx), y(sy), walls(std::move(ws)),
        hazards(std::move(hs)) {
    if (width < 4 || width > 10 || height < 4 || height > 10 ||
        max_turns < 1 || max_turns > 100)
      throw std::invalid_argument("invalid grid dimensions or turn limit");
    auto inside = [&](int px, int py) {
      return px >= 0 && px < width && py >= 0 && py < height;
    };
    if (!inside(start_x, start_y) || !inside(goal_x, goal_y) ||
        (start_x == goal_x && start_y == goal_y))
      throw std::invalid_argument("invalid grid start or goal");
    std::set<std::pair<int, int>> occupied;
    for (const auto& point : walls) {
      if (!inside(point.first, point.second) ||
          !occupied.insert(point).second)
        throw std::invalid_argument("invalid or duplicate grid wall");
    }
    for (const auto& point : hazards) {
      if (!inside(point.first, point.second) ||
          !occupied.insert(point).second)
        throw std::invalid_argument("invalid or overlapping grid hazard");
    }
    if (occupied.count({start_x, start_y}) || occupied.count({goal_x, goal_y}))
      throw std::invalid_argument("grid start and goal cannot be blocked");
  }

  static GridWorld from_file(const char* path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("cannot open grid specification: ") + path);
    int w, h, max_t, sx, sy, gx, gy, count;
    if (!(input >> w >> h >> max_t >> sx >> sy >> gx >> gy))
      throw std::runtime_error("invalid grid specification header");
    auto read_points = [&](std::vector<std::pair<int, int>>& points) {
      if (!(input >> count) || count < 0 || count > 100)
        throw std::runtime_error("invalid grid coordinate count");
      for (int i = 0; i < count; i++) {
        int px, py;
        if (!(input >> px >> py)) throw std::runtime_error("invalid grid coordinate");
        points.emplace_back(px, py);
      }
    };
    std::vector<std::pair<int, int>> ws, hs;
    read_points(ws);
    read_points(hs);
    std::string extra;
    if (input >> extra) throw std::runtime_error("unexpected data in grid specification");
    return GridWorld(w, h, max_t, sx, sy, gx, gy, std::move(ws), std::move(hs));
  }

  bool per_action() const override { return true; }
  int n_state() const override { return 13; }
  int n_actions() const override { return 4; }

  void reset(Rng&) override {
    x = start_x;
    y = start_y;
    turns = 0;
  }

  int legal(int* out) override {
    static const int dx[] = {0, 0, -1, 1};
    static const int dy[] = {-1, 1, 0, 0};
    int n = 0;
    for (int action = 0; action < 4; action++) {
      int nx = x + dx[action], ny = y + dy[action];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height ||
          std::find(walls.begin(), walls.end(), std::make_pair(nx, ny)) != walls.end())
        continue;
      out[n++] = action;
    }
    return n;
  }

  void state_feat(float* out) override {
    static const int dx[] = {0, 0, -1, 1};
    static const int dy[] = {-1, 1, 0, 0};
    out[0] = (float)x / (width - 1);
    out[1] = (float)y / (height - 1);
    out[2] = (float)goal_x / (width - 1);
    out[3] = (float)goal_y / (height - 1);
    out[4] = (float)(max_turns - turns) / max_turns;
    for (int action = 0; action < 4; action++) {
      int nx = x + dx[action], ny = y + dy[action];
      bool inside = nx >= 0 && nx < width && ny >= 0 && ny < height;
      out[5 + action * 2] =
          !inside || std::find(walls.begin(), walls.end(), std::make_pair(nx, ny)) != walls.end();
      out[6 + action * 2] =
          inside && std::find(hazards.begin(), hazards.end(), std::make_pair(nx, ny)) != hazards.end();
    }
  }

  float step(int action, bool& done, Rng&) override {
    int actions[4];
    int count = legal(actions);
    if (std::find(actions, actions + count, action) == actions + count)
      throw std::invalid_argument("illegal grid action");
    static const int dx[] = {0, 0, -1, 1};
    static const int dy[] = {-1, 1, 0, 0};
    x += dx[action];
    y += dy[action];
    turns++;
    if (std::find(hazards.begin(), hazards.end(), std::make_pair(x, y)) != hazards.end()) {
      done = true;
      return -1.f;
    }
    if (x == goal_x && y == goal_y) {
      done = true;
      return 1.f;
    }
    if (turns >= max_turns) {
      done = true;
      return -.25f;
    }
    done = false;
    return -.01f;
  }

  std::unique_ptr<Env> make() const override {
    return std::make_unique<GridWorld>(*this);
  }
};
