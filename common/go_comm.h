#pragma once

#include <inttypes.h>
#include <string>
#include <vector>

// Return code of functions should be "int"
typedef uint8_t GoStoneColor; // Stone color
typedef int16_t GoCoordId;    // Stone IDs or coordinates
typedef int16_t GoBlockId;    // Block IDs
typedef int16_t GoSize;       // Counts of visit times, used blocks, ..

namespace GoComm {

const GoCoordId BOARD_SIZE = 19;
const GoCoordId BOARD_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE;

const float KOMI = 7.5;

const GoCoordId COORD_UNSET = -2;
const GoCoordId COORD_PASS = -1;
const GoCoordId COORD_RESIGN = -3;

const GoSize SIZE_NONE = 0;

const GoBlockId MAX_BLOCK_SIZE = 1 << 8;
const GoBlockId BLOCK_UNSET = -1;

const GoStoneColor EMPTY = 0;
const GoStoneColor BLACK = 1;
const GoStoneColor WHITE = 2;
const GoStoneColor WALL = 3;
const GoStoneColor COLOR_UNKNOWN = -1;
const char *const COLOR_STRING[] = {"Empty", "Black", "White", "Wall"};

const GoCoordId DELTA_X[] = {0, 1, 0, -1};
const GoCoordId DELTA_Y[] = {-1, 0, 1, 0};
const GoSize DELTA_SIZE = sizeof(DELTA_X) / sizeof(*DELTA_X);

const GoSize UINT64_BITS = sizeof(uint64_t) * 8;
const GoSize LIBERTY_STATE_SIZE = (BOARD_INTERSECTIONS + UINT64_BITS - 1) / UINT64_BITS;
const GoSize BOARD_STATE_SIZE = (BOARD_INTERSECTIONS + UINT64_BITS - 1) / UINT64_BITS;

} // namespace GoComm

namespace GoFeature {

const int SIZE_HISTORYEACHSIDE = 8;
const int SIZE_PLAYERCOLOR = 2;

const int FEATURE_COUNT = 2 * SIZE_HISTORYEACHSIDE + SIZE_PLAYERCOLOR;

} // namespace GoFeature

namespace GoFunction {

extern bool InBoard(const GoCoordId id);

extern bool InBoard(const GoCoordId x, const GoCoordId y);

extern bool IsPass(const GoCoordId id);

extern bool IsPass(const GoCoordId x, const GoCoordId y);

extern bool IsUnset(const GoCoordId id);

extern bool IsUnset(const GoCoordId x, const GoCoordId y);

extern bool IsResign(const GoCoordId id);

extern bool IsResign(const GoCoordId x, const GoCoordId y);

extern void IdToCoord(const GoCoordId id, GoCoordId &x, GoCoordId &y);

extern GoCoordId CoordToId(const GoCoordId x, const GoCoordId y);

extern void SgfMoveToCoord(const std::string &str, GoCoordId &x, GoCoordId &y);

extern std::string CoordToSgfMove(const GoCoordId x, const GoCoordId y);

extern std::string CoordToBoardMove(const GoCoordId x, const GoCoordId y);

extern std::string IdToSgfMove(const GoCoordId id);

extern std::string IdToBoardMove(const GoCoordId id);

extern GoCoordId SgfMoveToId(const std::string &str);

extern void CreateGlobalVariables();

extern void CreateHashWeights();

extern void CreateNeighbourCache();

extern void CreateQuickLog2Table();

extern void CreateZobristHash();

} // namespace GoFunction

typedef std::pair<GoCoordId, GoCoordId> GoPosition;
typedef std::pair<uint64_t, uint64_t> GoHashValuePair;

extern GoHashValuePair g_hash_weight[GoComm::BOARD_SIZE][GoComm::BOARD_SIZE];
const GoHashValuePair g_hash_unit(3, 7);
extern uint64_t g_zobrist_board_hash_weight[4][GoComm::BOARD_INTERSECTIONS];
extern uint64_t g_zobrist_ko_hash_weight[GoComm::BOARD_INTERSECTIONS];
extern uint64_t g_zobrist_player_hash_weight[4];

extern std::vector<GoPosition> g_neighbour_cache_by_coord[GoComm::BOARD_SIZE]
                                                         [GoComm::BOARD_SIZE];
extern GoSize g_neighbour_size[GoComm::BOARD_INTERSECTIONS];
extern GoCoordId g_neighbour_cache_by_id[GoComm::BOARD_INTERSECTIONS]
                                        [GoComm::DELTA_SIZE + 1];
extern GoCoordId g_log2_table[67];

#define FOR_NEI(id, nb)                                                        \
  for (GoCoordId *nb = g_neighbour_cache_by_id[(id)];                          \
       GoComm::COORD_UNSET != *nb; ++nb)
#define FOR_EACHCOORD(id)                                                      \
  for (GoCoordId id = 0; id < GoComm::BOARD_INTERSECTIONS; ++id)
#define FOR_EACHBLOCK(id)                                                      \
  for (GoBlockId id = 0; id < GoComm::MAX_BLOCK_SIZE; ++id)
