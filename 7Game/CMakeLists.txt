cmake_minimum_required(VERSION 3.10)
project(SevensCardGame VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(SOURCES 
    src/Card.cpp
    src/Deck.cpp
    src/Hand.cpp
    src/PlacedSuit.cpp
    src/SevensGame.cpp
    src/ComputerPlayer.cpp
    src/HumanPlayerCLI.cpp
    src/SevensGameCLI.cpp
    src/main.cpp
)

add_executable(sevens_game ${SOURCES})
target_include_directories(sevens_game PRIVATE ${INCLUDE_DIRS})
