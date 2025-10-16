class ClusterAnalyzer:
    """Analyze cluster profiles and generate recommendations"""
    
    def analyze_cluster_profiles(self, data, labels, feature_names):
        """Analisis profil setiap cluster"""
        df_analysis = data.copy()
        df_analysis['Cluster'] = labels
        
        # Add IPK if not exists
        if 'IPK_Skala' not in df_analysis.columns:
            ipk_map = {'> 3.5': 4, '3.00 - 3.50': 3, '2.51 - 3.00': 2, 
                      '2.01 - 2.50': 1, '< 2.00': 0}
            df_analysis['IPK_Skala'] = df_analysis['Indeks Prestasi Kumulatif (IPK) Terakhir'].map(ipk_map)
        
        # Profile analysis
        profile_cols = feature_names + ['IPK_Skala']
        cluster_profiles = df_analysis.groupby('Cluster')[profile_cols].mean()
        
        # Interpretation
        interpretations = []
        for cluster in cluster_profiles.index:
            profile = cluster_profiles.loc[cluster]
            
            # Learning style analysis
            # Deep Learning Indicators
            deep_learning_score = (
                profile.get('Saya sering mencari sumber belajar (jurnal, buku, video) di luar materi yang diberikan oleh dosen.', 0) +
                profile.get('Saya suka mencari tahu hubungan antara materi kuliah dengan isu-isu di dunia nyata atau bidang ilmu lain.', 0) +
                profile.get('Saya baru merasa benar-benar paham suatu topik jika saya bisa menjelaskannya kepada orang lain, bukan hanya bisa mengerjakan soalnya.', 0) +
                profile.get('Jika menemukan konsep yang rumit, saya tertantang untuk memahaminya lebih dalam, meskipun itu memakan banyak waktu.', 0)
            ) / 4
            
            # Surface Learning Indicators
            surface_learning_score = (
                profile.get('Tujuan utama saya belajar adalah untuk mendapatkan nilai ujian yang tinggi.', 0) +
                profile.get('Bagi saya, membaca ulang slide presentasi dan catatan dari dosen sudah lebih dari cukup untuk persiapan ujian.', 0)
            ) / 2
            
            # Strategic Learning Indicators
            strategic_learning_score = (
                profile.get('Saya merasa adanya tekanan deadline membuat saya lebih fokus dan termotivasi.', 0) +
                profile.get('Saya cenderung mempelajari materi secara bertahap sepanjang semester, bukan hanya saat mendekati ujian (Sistem Kebut Semalam).', 0) +
                profile.get('Saya secara aktif menggunakan umpan balik (feedback) dari dosen pada tugas untuk memperbaiki pemahaman konseptual saya, bukan hanya untuk memperbaiki nilai.', 0)
            ) / 3
            
            # Determine dominant learning style
            learning_styles = {
                'Deep Learner': deep_learning_score,
                'Surface Learner': surface_learning_score,
                'Strategic Learner': strategic_learning_score
            }
            dominant_style = max(learning_styles, key=learning_styles.get)
            
            interpretation = {
                'cluster': cluster,
                'size': len(df_analysis[df_analysis['Cluster'] == cluster]),
                'ipk': profile.get('IPK_Skala', 0),
                'deep_learning': deep_learning_score,
                'surface_learning': surface_learning_score,
                'strategic_learning': strategic_learning_score,
                'dominant_style': dominant_style,
                'recommendations': self._generate_recommendations(dominant_style, profile)
            }
            interpretations.append(interpretation)
        
        return {
            'profiles': cluster_profiles,
            'interpretations': interpretations,
            'data_with_clusters': df_analysis
        }

    def _generate_recommendations(self, learning_style, profile):
        """Generate recommendations based on learning style"""
        recommendations = []
        
        if learning_style == 'Deep Learner':
            recommendations.extend([
                "Pertahankan gaya belajar mendalam Anda",
                "Terus eksplorasi materi di luar kurikulum",
                "Manfaatkan kemampuan menjelaskan konsep untuk membantu teman",
                "Coba aplikasikan pengetahuan dalam proyek nyata"
            ])
        elif learning_style == 'Surface Learner':
            recommendations.extend([
                "Coba fokus pada pemahaman konsep daripada sekadar nilai",
                "Eksplorasi hubungan antara materi kuliah dengan aplikasi nyata",
                "Bergabung dengan kelompok diskusi untuk memperdalam pemahaman",
                "Setel tujuan belajar yang lebih dari sekadar nilai ujian"
            ])
        else:  # Strategic Learner
            recommendations.extend([
                "Manfaatkan kemampuan manajemen waktu yang baik",
                "Kombinasikan dengan pendekatan deep learning untuk hasil optimal",
                "Terus gunakan feedback untuk improvement",
                "Eksplorasi metode belajar yang lebih variatif"
            ])
        
        # Additional recommendations based on specific scores
        if profile.get('Saya sering mencari sumber belajar (jurnal, buku, video) di luar materi yang diberikan oleh dosen.', 0) < 3:
            recommendations.append("Coba eksplorasi sumber belajar tambahan di luar materi kuliah")
        
        if profile.get('Saya cenderung mempelajari materi secara bertahap sepanjang semester, bukan hanya saat mendekati ujian (Sistem Kebut Semalam).', 0) < 3:
            recommendations.append("Kembangkan kebiasaan belajar konsisten sepanjang semester")
        
        return recommendations
