#include "go_comm.h"

#include <cstring>
#include <mutex>
// #include <glog/logging.h>

#define x first
#define y second

using namespace std;
using namespace GoComm;

GoHashValuePair g_hash_weight[BOARD_SIZE][BOARD_SIZE];

vector<GoPosition> g_neighbour_cache_by_coord[BOARD_SIZE][BOARD_SIZE];
GoSize g_neighbour_size[BOARD_INTERSECTIONS];
GoCoordId g_neighbour_cache_by_id[BOARD_INTERSECTIONS][5];
GoCoordId g_log2_table[67];
uint64_t g_zobrist_board_hash_weight[4][BOARD_INTERSECTIONS];
uint64_t g_zobrist_ko_hash_weight[BOARD_INTERSECTIONS];
uint64_t g_zobrist_player_hash_weight[4];

namespace GoFunction {

bool InBoard(const GoCoordId id) {
  return 0 <= id && id < BOARD_INTERSECTIONS;
}

bool InBoard(const GoCoordId x, const GoCoordId y) {
  return 0 <= x && x < BOARD_SIZE && 0 <= y && y < BOARD_SIZE;
}

bool IsPass(const GoCoordId id) {
  return COORD_PASS == id;
}

bool IsPass(const GoCoordId x, const GoCoordId y) {
  return COORD_PASS == CoordToId(x, y);
}

bool IsUnset(const GoCoordId id) {
  return COORD_UNSET == id;
}

bool IsUnset(const GoCoordId x, const GoCoordId y) {
  return COORD_UNSET == CoordToId(x, y);
}

bool IsResign(const GoCoordId id) {
  return COORD_RESIGN == id;
}

bool IsResign(const GoCoordId x, const GoCoordId y) {
  return COORD_RESIGN == CoordToId(x, y);
}

void IdToCoord(const GoCoordId id, GoCoordId &x, GoCoordId &y) {
  if (COORD_PASS == id) {
    x = y = COORD_PASS;
  } else if (COORD_RESIGN == id) {
    x = y = COORD_RESIGN;
  } else if (!InBoard(id)) {
    x = y = COORD_UNSET;
  } else {
    x = id / BOARD_SIZE;
    y = id % BOARD_SIZE;
  }
}

GoCoordId CoordToId(const GoCoordId x, const GoCoordId y) {
  if (COORD_PASS == x && COORD_PASS == y) {
    return COORD_PASS;
  }
  if (COORD_RESIGN == x && COORD_RESIGN == y) {
    return COORD_RESIGN;
  }
  if (!InBoard(x, y)) {
    return COORD_UNSET;
  }
  return x * BOARD_SIZE + y;
}

void SgfMoveToCoord(const string &str, GoCoordId &x, GoCoordId &y) {
  // CHECK_EQ(str.length(), 2) << "string[" << str << "] length not equal to 2";
  x = str[0] - 'a';
  y = str[1] - 'a';
  if (str == "zz") {
    x = y = COORD_PASS;
  } else if (!InBoard(x, y)) {
    x = y = COORD_UNSET;
  }
}

string CoordToSgfMove(const GoCoordId x, const GoCoordId y) {
    char buffer[3];
    if (!InBoard(x, y)) {
        buffer[0] = buffer[1] = 'z';
    } else {
        buffer[0] = x + 'a';
        buffer[1] = y + 'a';
    }
    return string(buffer, 2);
}

string CoordToBoardMove(const GoCoordId x, const GoCoordId y) {
  if (!InBoard(x, y)) {
    return "pass";
  } else {
    return std::string({x > 7 ? char('B' + x) : char('A' + x)}) + std::to_string(BOARD_SIZE - y);
  }
}

std::string IdToSgfMove(const GoCoordId id) {
  GoCoordId x, y;
  IdToCoord(id, x, y);
  return CoordToSgfMove(x, y);
}

std::string IdToBoardMove(const GoCoordId id) {
  GoCoordId x, y;
  IdToCoord(id, x, y);
  return CoordToBoardMove(x, y);
}

GoCoordId SgfMoveToId(const std::string &str) {
  GoCoordId x, y;
  SgfMoveToCoord(str, x, y);
  return CoordToId(x, y);
}

once_flag CreateGlobalVariables_once;
void CreateGlobalVariables() {
  call_once(CreateGlobalVariables_once, []() {
    CreateNeighbourCache();
    CreateHashWeights();
    CreateQuickLog2Table();
    CreateZobristHash();
  });
}

void CreateHashWeights() {
  g_hash_weight[0][0] = GoHashValuePair(1, 1);
  for (GoCoordId i = 1; i < BOARD_INTERSECTIONS; ++i) {
    g_hash_weight[i / BOARD_SIZE][i % BOARD_SIZE] = GoHashValuePair(
        g_hash_weight[(i - 1) / BOARD_SIZE][(i - 1) % BOARD_SIZE].x * g_hash_unit.x,
        g_hash_weight[(i - 1) / BOARD_SIZE][(i - 1) % BOARD_SIZE].y * g_hash_unit.y);
  }
}

void CreateNeighbourCache() {
  for (GoCoordId x = 0; x < BOARD_SIZE; ++x) {
    for (GoCoordId y = 0; y < BOARD_SIZE; ++y) {
      GoCoordId id = CoordToId(x, y);

      g_neighbour_cache_by_coord[x][y].clear();
      for (int i = 0; i <= DELTA_SIZE; ++i) {
        g_neighbour_cache_by_id[id][i] = COORD_UNSET;
      }
      for (int i = 0; i < DELTA_SIZE; ++i) {
        GoCoordId nx = x + DELTA_X[i];
        GoCoordId ny = y + DELTA_Y[i];

        if (InBoard(nx, ny)) {
          g_neighbour_cache_by_coord[x][y].push_back(GoPosition(nx, ny));
        }
      }
      g_neighbour_size[id] = g_neighbour_cache_by_coord[x][y].size();
      for (GoSize i = 0; i < g_neighbour_cache_by_coord[x][y].size(); ++i) {
        g_neighbour_cache_by_id[id][i] = CoordToId(
            g_neighbour_cache_by_coord[x][y][i].x,
            g_neighbour_cache_by_coord[x][y][i].y);
      }
    }
  }
  // cerr << hex << int(g_neighbour_cache_by_coord) << endl;
}

void CreateQuickLog2Table() {
  memset(g_log2_table, -1, sizeof(g_log2_table));
  int tmp = 1;

  for (GoCoordId i = 0; i < 64; ++i) {
    g_log2_table[tmp] = i;
    tmp *= 2;
    tmp %= 67;
  }
}

void CreateZobristHash() {
  uint32_t seed = 0xdeadbeaf;

  for (int i = 0; i < 4; ++i) {
    g_zobrist_player_hash_weight[i] = (uint64_t)rand_r(&seed) << 32 | rand_r(&seed);
    for (int j = 0; j < BOARD_INTERSECTIONS; ++j) {
      g_zobrist_board_hash_weight[i][j] = (uint64_t)rand_r(&seed) << 32 | rand_r(&seed);
    }
  }

  for (int i = 0; i < BOARD_INTERSECTIONS; ++i) {
    g_zobrist_ko_hash_weight[i] = (uint64_t)rand_r(&seed) << 32 | rand_r(&seed);
  }
}

} // namespace GoFunction

#undef y
#undef x
