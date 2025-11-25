import inference
from ultralytics import YOLO
import time, statistics, os
import plotly.express as px
import psycopg2

# -----------  БАЗА ДАННЫХ  -----------
DB_NAME = 'postgres'
DB_USER = 'postgres'
DB_PASSWORD = '1234'
DB_HOST = 'localhost'
DB_PORT = '5432'

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS tomatoes(
    id      SERIAL PRIMARY KEY,
    weight  INTEGER
)''')
conn.commit()
# -------------------------------------

model = YOLO('runs/detect/train2/weights/best.pt')
model.fuse()

path = input('Укажите путь к необработанному видео: ').strip()
if path.startswith('"') and path.endswith('"'):
    path = path[1:-1]

# обрабатываем видео (возвращает dict {id: вес})
# после обработки ролика
weights_dict = inference.process_video_with_tracking(
    model, path, show_video=True, save_video=False, output_video_path='output_video2.mp4'
)

# weights_dict сейчас – dict_values; превращаем в настоящий dict
# если в inference последняя строка была  return weights.values()
# то достаточно так:
weights_dict = dict(enumerate(weights_dict))   # ID – порядковые номера 0,1,2...

# или, если ID томатов всё-таки важны, нужно изменить inference.py
# чтобы он возвращал сам словарь weights, а не weights.values()

# -----------  ПИШЕМ В БД  -----------
for tomato_id, weight in weights_dict.items():
    cur.execute('INSERT INTO tomatoes(id, weight) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET weight = EXCLUDED.weight', (tomato_id, weight))
# ------------------------------------

# -----------  СТАТИСТИКА  -----------
data = list(weights_dict.values())          # список весов
print('Отсортированные веса:', sorted(data))

fig_hist = px.histogram(
    data, nbins=20, title='Результат обработки видео',
    labels={'variable': 'Полученные данные'},
    color_discrete_sequence=['#00A86B']
).update_xaxes(categoryorder='total ascending')

fig_hist.update_layout(
    xaxis_title='Вес томатов, г.',
    yaxis_title='Количество томатов, шт.',
    bargap=0.1,
    showlegend=False,
    legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.8, itemwidth=100),
    plot_bgcolor='#e2eeee'
)

fig_hist.add_annotation(
    text='Полученные данные:', showarrow=False,
    xref='paper', yref='paper', x=0.95, y=0.97,
    height=20, width=300, bgcolor='white', font=dict(weight=500, size=14)
)
fig_hist.add_annotation(
    text=f'Общее количество томатов, шт: {len(data)}<br>'
         f'Вес, кг: {sum(data)//1000}<br>'
         f'Среднее значение, г: {sum(data)//len(data)}<br>'
         f'Мода, г: {statistics.mode(data)}<br>'
         f'Максимальное значение, г: {max(data)}<br>'
         f'Минимальное значение, г: {min(data)}',
    align='left', showarrow=False,
    xref='paper', yref='paper', x=0.95, y=0.93,
    height=120, width=300, bgcolor='white'
)

fig_hist.for_each_trace(lambda t: t.update(name='Томаты',
                                           legendgroup='Томаты',
                                           hovertemplate=t.hovertemplate.replace(t.name, 'Томаты')))
fig_hist.show()

# закрываем соединение с БД
conn.close()