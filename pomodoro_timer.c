#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#include <termios.h>
#include <fcntl.h>
#include <locale.h>

#define WORK_TIME 25
#define SHORT_BREAK 5
#define LONG_BREAK 15
#define SESSIONS_BEFORE_LONG_BREAK 4

// 전역 변수
int work_time = WORK_TIME;
int short_break = SHORT_BREAK;
int long_break = LONG_BREAK;
int sessions_before_long_break = SESSIONS_BEFORE_LONG_BREAK;
int remaining_time = WORK_TIME;
int is_paused = 0;
int current_session = 1;
int total_sessions = 0;
int total_work_time = 0;
char current_task[100] = "";

// 터미널 설정 구조체
struct termios orig_termios;

// 터미널 설정 초기화
void init_terminal() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    struct termios new_termios = orig_termios;
    new_termios.c_lflag &= ~ICANON;  // ECHO는 유지
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
}

// 터미널 설정 복원
void restore_terminal() {
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
}

// 진행률 표시 함수
void display_progress(int current, int total, int width) {
    float progress = (float)current / total;
    int filled = (int)(progress * width);
    
    printf("\r[");
    for (int i = 0; i < width; i++) {
        if (i < filled) printf("=");
        else printf(" ");
    }
    printf("] %d%%", (int)(progress * 100));
    fflush(stdout);
}

// 타이머 표시 함수
void display_timer(int minutes, int seconds, int total_seconds, const char* phase) {
    printf("\033[2J\033[H");  // 화면 지우기
    printf("=== 뽀모도로 타이머 ===\n\n");
    
    if (strlen(current_task) > 0) {
        printf("현재 작업: %s\n\n", current_task);
    }
    
    printf("단계: %s\n", phase);
    printf("남은 시간: %02d:%02d\n", minutes, seconds);
    
    // 진행률 표시
    display_progress(total_seconds - (minutes * 60 + seconds), total_seconds, 50);
    
    printf("\n\n");
    printf("p: 일시정지/재개\n");
    printf("q: 종료\n");
    fflush(stdout);
}

// 세션 기록 저장 함수
void save_session_record(const char* phase, int duration) {
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char filename[64];
    sprintf(filename, "pomodoro_log_%04d%02d%02d.txt", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
    
    FILE* file = fopen(filename, "a");
    if (file) {
        char task_info[120] = "";
        if (strlen(current_task) > 0) {
            sprintf(task_info, " (작업: %s)", current_task);
        }
        fprintf(file, "[%02d:%02d:%02d] %s 완료(소요 시간: %d분)%s\n", 
                t->tm_hour, t->tm_min, t->tm_sec, phase, duration, task_info);
        fclose(file);
    }
}

// 타이머 실행 함수
void run_timer(int duration, const char* phase) {
    int total_seconds = duration * 60;
    int remaining_seconds = total_seconds;
    int paused = 0;
    time_t start_time = time(NULL);
    time_t pause_start = 0;
    int total_pause_time = 0;
    
    // 타이머 실행 중에는 ECHO를 끄고
    struct termios timer_termios;
    tcgetattr(STDIN_FILENO, &timer_termios);
    timer_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &timer_termios);
    
    while (remaining_seconds > 0) {
        if (!paused) {
            time_t current_time = time(NULL);
            int elapsed = current_time - start_time - total_pause_time;
            remaining_seconds = total_seconds - elapsed;
            
            if (remaining_seconds < 0) remaining_seconds = 0;
            
            int minutes = remaining_seconds / 60;
            int seconds = remaining_seconds % 60;
            
            display_timer(minutes, seconds, total_seconds, phase);
        }
        
        // 키 입력 확인
        if (kbhit()) {
            char key = getchar();
            if (key == 'p' || key == 'P') {
                if (!paused) {
                    paused = 1;
                    pause_start = time(NULL);
                    printf("\n일시정지 중... (p를 눌러 재개)\n");
                } else {
                    paused = 0;
                    total_pause_time += time(NULL) - pause_start;
                }
            } else if (key == 'q' || key == 'Q') {
                printf("\n타이머를 종료합니다.\n");
                // 타이머 종료 시 원래 터미널 설정으로 복원
                tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
                return;
            }
        }
        
        usleep(100000);  // 0.1초 대기
    }
    
    // 타이머 종료 시 원래 터미널 설정으로 복원
    tcsetattr(STDIN_FILENO, TCSANOW, &orig_termios);
    save_session_record(phase, duration);
}

// 설정 메뉴 함수
void settings_menu() {
    while (1) {
        printf("\033[2J\033[H");  // 화면 지우기
        printf("=== 설정 ===\n\n");
        printf("1. 작업 시간 설정 (현재: %d분)\n", work_time);
        printf("2. 짧은 휴식 시간 설정 (현재: %d분)\n", short_break);
        printf("3. 긴 휴식 시간 설정 (현재: %d분)\n", long_break);
        printf("4. 작업 이름 설정 (현재: %s)\n", strlen(current_task) > 0 ? current_task : "없음");
        printf("5. 기본값으로 복원\n");
        printf("0. 돌아가기\n\n");
        printf("선택: ");
        
        int choice;
        scanf("%d", &choice);
        getchar();  // 개행 문자 제거
        
        switch (choice) {
            case 1:
                printf("작업 시간(분): ");
                scanf("%d", &work_time);
                getchar();  // 개행 문자 제거
                break;
            case 2:
                printf("짧은 휴식 시간(분): ");
                scanf("%d", &short_break);
                getchar();  // 개행 문자 제거
                break;
            case 3:
                printf("긴 휴식 시간(분): ");
                scanf("%d", &long_break);
                getchar();  // 개행 문자 제거
                break;
            case 4:
                printf("작업 이름: ");
                fgets(current_task, sizeof(current_task), stdin);
                current_task[strcspn(current_task, "\n")] = 0;
                break;
            case 5:
                work_time = WORK_TIME;
                short_break = SHORT_BREAK;
                long_break = LONG_BREAK;
                current_task[0] = '\0';
                printf("기본값으로 복원되었습니다.\n");
                sleep(1);
                break;
            case 0:
                return;
        }
    }
}

// 통계 메뉴 함수
void statistics_menu() {
    printf("\033[2J\033[H");  // 화면 지우기
    printf("=== 통계 ===\n\n");
    
    time_t now = time(NULL);
    struct tm* t = localtime(&now);
    char filename[32];
    sprintf(filename, "pomodoro_log_%04d%02d%02d.txt", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday);
    
    FILE* file = fopen(filename, "r");
    if (file) {
        char line[256];
        int total_work_time = 0;
        int total_breaks = 0;
        int sessions = 0;
        
        while (fgets(line, sizeof(line), file)) {
            int duration = 0;
            char* dur = strstr(line, "소요 시간: ");
            if (dur) {
                dur += strlen("소요 시간: ");
                sscanf(dur, "%d", &duration);
                
                if (strstr(line, "작업") || strstr(line, "Work")) {
                    total_work_time += duration;
                    sessions++;
                } else if (strstr(line, "휴식") || strstr(line, "Break")) {
                    total_breaks += duration;
                }
            }
        }
        
        printf("오늘의 작업 기록:\n");
        printf("- 완료한 세션: %d개\n", sessions);
        printf("- 총 작업 시간: %d분\n", total_work_time);
        printf("- 총 휴식 시간: %d분\n", total_breaks);
        
        fclose(file);
    } else {
        printf("오늘의 작업 기록이 없습니다.\n");
    }
    
    printf("\nEnter를 눌러 돌아가기...");
    getchar();
}

// 메인 메뉴 함수
void main_menu() {
    int completed_sessions = 0;
    
    while (1) {
        printf("\033[2J\033[H");  // 화면 지우기
        printf("=== 뽀모도로 타이머 ===\n\n");
        printf("1. 타이머 시작\n");
        printf("2. 설정\n");
        printf("3. 통계\n");
        printf("0. 종료\n\n");
        printf("선택: ");
        
        int choice;
        scanf("%d", &choice);
        getchar();  // 개행 문자 제거
        
        switch (choice) {
            case 1:
                run_timer(work_time, "작업");
                completed_sessions++;
                
                if (completed_sessions % sessions_before_long_break == 0) {
                    run_timer(long_break, "긴 휴식");
                } else {
                    run_timer(short_break, "짧은 휴식");
                }
                break;
            case 2:
                settings_menu();
                break;
            case 3:
                statistics_menu();
                break;
            case 0:
                return;
        }
    }
}

int main() {
    setlocale(LC_ALL, "ko_KR.UTF-8");
    init_terminal();
    main_menu();
    restore_terminal();
    return 0;
}

// kbhit 함수 구현
int kbhit() {
    struct termios oldt, newt;
    int ch;
    int oldf;
    
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    
    ch = getchar();
    
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    
    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    
    return 0;
} 